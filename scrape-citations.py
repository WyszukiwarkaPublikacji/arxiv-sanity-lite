import requests
import os
import shutil
import bibtexparser
import random
from pathlib import Path
from aslite.db import get_papers_db
import sys
from json import dumps
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sqlalchemy import insert, table, select
from db.SQLLite.OrmDB import Citations
from sqlalchemy.orm import Session
from db.SQLLite.OrmDB import Papers
from db.SQLLiteAlchemyInstance import SQLAlchemyInstance
from db.Milvus.MilvusSetterDB import MilvusSetterDB
from db.Milvus.MilvusMetaRepository import MilvusMetaRepository



def find_files(folder_path: str, ext: str):
    print(Path(folder_path))
    return list(Path(folder_path).rglob(f'*{ext}'))


def clear_directory(target_dir):
    os.rmtree(target_dir)


def create_dir(name):
    os.mkdir(name)


def download_arxiv_source(arxiv_id: str):
    url = f"https://arxiv.org/src/{arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(f"{arxiv_id}.tar.gz", "wb") as f:
        print(str(response.content))
        f.write(response.content)


def extract_citations_from_latex(latex_text):

    pattern = r'\\cite\w*(?:\[[^\]]*\])?\{([^}]+)\}'
    matches = re.findall(pattern, latex_text)    

    citation_keys = []
    for match in matches:
        keys = [key.strip() for key in match.split(',')]
        citation_keys.extend(keys)
    
    return citation_keys


def extract_tar(path, extract_to):
    os.system(f"tar -xzf {path} -C {extract_to}")
    os.remove(path)


def parse_bibtex_files(bib_files):
    entries = []
    for bib_file in bib_files:
        try:
            with bib_file.open(encoding='utf-8') as bibtex_file:                
                bib_database = bibtexparser.load(bibtex_file)
                entries.extend(bib_database.entries)
        except Exception as e:
            print(f"Error parsing {bib_file}: {e}", file=sys.stderr)

    entries_dict = {}
    for entry in entries:
        entries_dict |= {entry["ID"]: entry}
    
    return entries_dict


def extract_arxiv_from_entries(entries, tex_citations):
    pattern = r"(?:\b(?:\d{4}\.\d{4,5})(?:v\d+)?\b)|(?:\b[a-z\-]+(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?\b)"

    arxiv = []
    for citation in tex_citations:
        try:
            entry = str(entries[citation])
            match = re.findall(pattern, entry)
            if match and "arxiv" in str(entries[citation]).lower():
                arxiv += [match[0]]

        except KeyError:
            pass
    return arxiv


def internet_part(ID):
    download_arxiv_source(ID)


def cpu_part(ID):
    print("CPU START", ID)

    create_dir(ID)
    extract_tar(f"{ID}.tar.gz", ID)

    tex_files = find_files(ID, ".tex")
    bib_files = find_files(ID, ".bib")


    text = ""
    for tex_file in tex_files:
        with open(tex_file, 'r') as f:
            text += f.read()
    
    citation_keys = extract_citations_from_latex(text)

    sstderr = sys.stderr
    sys.stderr = open('trash', 'w')
    entries = parse_bibtex_files(bib_files)
    sys.stderr = sstderr

    arxiv = extract_arxiv_from_entries(entries, citation_keys)
    clear_directory(ID)


    return arxiv


def scrape(ID):
    internet_part(ID)
    return cpu_part(ID)


async def handle_key(key, io_executor, cpu_executor):
    loop = asyncio.get_running_loop()

    # Step 1: Download file (I/O)
    await loop.run_in_executor(io_executor, internet_part, key)
    print(f"[IO DONE] {key}")

    # Step 2: Process file (CPU)
    result = await loop.run_in_executor(cpu_executor, cpu_part, key)
    print(f"[CPU DONE] {key}: {result}")

    return key, result

async def compute():
    pdb = get_papers_db(flag='r')
    keys = list(pdb.keys())
    random.shuffle(keys)
    N = 1

    with ThreadPoolExecutor(max_workers=10) as io_pool, \
         ProcessPoolExecutor(max_workers=4) as cpu_pool:

        tasks = [handle_key(key, io_pool, cpu_pool) for key in keys[:N]]

        # Process keys as they complete
        results = {}
        for coro in asyncio.as_completed(tasks):
            key, result = await coro
            results[key] = result

    return results


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # macOS default

    return asyncio.run(compute())

if __name__ == '__main__':
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()

    MilvusSetterDB.create_collectio_metas()
    MilvusSetterDB.create_collection_papers()
    
    with Session(engine) as session:
        stmt = insert(Citations).values(id="3",origin_publication_id="username", citation_publication_id="Full Username")
        result = session.execute(stmt)
        repo = MilvusMetaRepository()

        for i in session.execute(select(Citations)):
            print(i)
       

