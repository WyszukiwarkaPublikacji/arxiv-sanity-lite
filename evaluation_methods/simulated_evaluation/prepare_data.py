import os
import random
import numpy as np
import torch
from aslite.db import CompressedSqliteDict
from aslite.env import DATA_DIR
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict
from evaluation_methods.simulated_evaluation.disk_dataset import DiskDataset

TEST_AUTHORS_PATH = os.path.join(DATA_DIR, "test_authors.txt")
DATA_META_PATH = os.path.join(DATA_DIR, "data_meta.npz")

X_TRAIN_PATH = os.path.join(DATA_DIR, "x_train.dat")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.dat")

X_TEST_PATH = os.path.join(DATA_DIR, "x_test.dat")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.dat")

def group_papers_by_authors(papers_db: CompressedSqliteDict) -> Tuple[List[str], Dict[str, List[str]]]:
    ids = []
    papers_by_authors = {}
    for arxiv_id, values in papers_db.items():
        if '.' not in arxiv_id:
            continue
        ids.append(arxiv_id)
        for author in values['authors']:
            papers_by_authors.setdefault(author['name'], []).append(arxiv_id)
    filtered_papers_by_authors = {}
    for author, papers in papers_by_authors.items():
        if len(papers) >= 3:
            filtered_papers_by_authors[author] = papers
    return ids, filtered_papers_by_authors

def pad_tensor(seq: torch.Tensor, target_length: int) -> torch.Tensor:
    if seq.shape[0] < target_length:
        pad = torch.full((target_length - seq.shape[0], seq.shape[1]), 0.0).to(seq.device)
        seq = torch.cat([pad, seq], dim=0)
    return seq

def process_author(author: str, papers: List[str], papers_db, ids: List[str], max_len: int) -> Tuple[List[torch.Tensor], List[float]]:
    sub_x, sub_y = [], []
    paper_vectors = [papers_db[paper]['paper_vector'] for paper in papers]
    papers_tensor = torch.tensor(np.array(paper_vectors))
    for idx in range(len(papers)):
        erased_vector = papers_tensor[idx].unsqueeze(dim=0)
        erased = torch.cat((papers_tensor[:idx], papers_tensor[idx+1:]))
        rnd_idx = random.randint(0, len(ids) - 1)
        rnd_tensor = torch.tensor(papers_db[ids[rnd_idx]]['paper_vector']).unsqueeze(dim=0)
        neg = torch.cat((erased, rnd_tensor))
        sub_x.append(pad_tensor(neg, max_len))
        sub_y.append(0.0)
        erased = torch.cat((erased, erased_vector))
        sub_x.append(pad_tensor(erased, max_len))
        sub_y.append(1.0)
    return sub_x, sub_y

def process_authors_to_disk(papers_by_authors: Dict[str, List[str]], papers_db, ids: List[str],
                            max_len: int, x_path: str, y_path: str, flush_interval: int = 5000) -> Tuple[int, Tuple, Tuple]:
    total_samples = sum(2 * len(papers) for papers in papers_by_authors.values())
    d = len(papers_db[ids[0]]['paper_vector'])
    x_shape = (total_samples, max_len, d)
    y_shape = (total_samples,)
    x_mem = np.memmap(x_path, mode='w+', shape=x_shape, dtype=np.float32)
    y_mem = np.memmap(y_path, mode='w+', shape=y_shape, dtype=np.float32)
    offsets = {}
    cur_offset = 0
    for author, papers in papers_by_authors.items():
        offsets[author] = cur_offset
        cur_offset += 2 * len(papers)
    def proc(author, papers):
        return author, *process_author(author, papers, papers_db, ids, max_len)
    writer_count = 0
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(proc, author, papers): author for author, papers in papers_by_authors.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing authors"):
            author, sub_x, sub_y = future.result()
            offset = offsets[author]
            for i in range(len(sub_y)):
                x_mem[offset + i] = sub_x[i].cpu().numpy().astype(np.float32)
                y_mem[offset + i] = np.float32(sub_y[i])
            writer_count += 1
            if writer_count % flush_interval == 0:
                x_mem.flush()
                y_mem.flush()
    x_mem.flush()
    y_mem.flush()
    return total_samples, x_shape, y_shape

def prepare_data(papers_db: CompressedSqliteDict) -> Tuple[Dataset, Dataset]:
    if all(os.path.exists(p) for p in [X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, DATA_META_PATH]):
        meta = np.load(DATA_META_PATH, allow_pickle=True)
        x_train_shape = tuple(meta["x_train_shape"])
        y_train_shape = tuple(meta["y_train_shape"])
        x_test_shape = tuple(meta["x_test_shape"])
        y_test_shape = tuple(meta["y_test_shape"])
        train_dataset = DiskDataset(X_TRAIN_PATH, Y_TRAIN_PATH, x_train_shape, y_train_shape)
        test_dataset = DiskDataset(X_TEST_PATH, Y_TEST_PATH, x_test_shape, y_test_shape)
    else:
        ids, papers_by_authors = group_papers_by_authors(papers_db)
        authors = list(papers_by_authors.keys())
        random.shuffle(authors)
        split = int(0.8 * len(authors))
        train_authors = set(authors[:split])
        papers_by_authors_train = {a: papers_by_authors[a] for a in train_authors}
        papers_by_authors_test = {a: papers_by_authors[a] for a in authors[split:]}
        with open(TEST_AUTHORS_PATH, "w") as f:
            for a in authors[split:]:
                f.write(a + "\n")
        max_len = max(len(papers) + 1 for papers in papers_by_authors.values())
        print(f"max_len={max_len}")
        total_train, x_train_shape, y_train_shape = process_authors_to_disk(papers_by_authors_train, papers_db, ids, max_len, X_TRAIN_PATH, Y_TRAIN_PATH)
        total_test, x_test_shape, y_test_shape = process_authors_to_disk(papers_by_authors_test, papers_db, ids, max_len, X_TEST_PATH, Y_TEST_PATH)
        np.savez(DATA_META_PATH,
                 x_train_shape=x_train_shape, y_train_shape=y_train_shape,
                 x_test_shape=x_test_shape, y_test_shape=y_test_shape,
                 total_train=total_train, total_test=total_test,
                 max_len=max_len)
        train_dataset = DiskDataset(X_TRAIN_PATH, Y_TRAIN_PATH, x_train_shape, y_train_shape)
        test_dataset = DiskDataset(X_TEST_PATH, Y_TEST_PATH, x_test_shape, y_test_shape)
    return train_dataset, test_dataset

def read_test_authors(path: str = TEST_AUTHORS_PATH) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f]
