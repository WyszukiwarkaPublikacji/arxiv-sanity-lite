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
from pymilvus import connections




def find_files(folder_path: str, ext: str):
    return list(Path(folder_path).rglob(f'*{ext}'))


def clear_directory(target_dir):
    shutil.rmtree(target_dir)


def create_dir(name):
    os.mkdir(name)


def download_arxiv_source(arxiv_id: str):
    url = f"https://arxiv.org/src/{arxiv_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(f"{arxiv_id}.tar.gz", "wb") as f:
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
    print("[DOWNLOADED]", ID)


def cpu_part(ID):

    create_dir(ID)
    extract_tar(f"{ID}.tar.gz", ID)
    print("[EXTRACTED]", ID)
    tex_files = find_files(ID, ".tex")
    bib_files = find_files(ID, ".bib")


    text = ""
    for tex_file in tex_files:
        with open(tex_file, 'r') as f:
            text += f.read()
    
    citation_keys = extract_citations_from_latex(text)
    print("[KEYS FOUND]", ID)
    sstderr = sys.stderr
    sys.stderr = open('trash', 'w')
    entries = parse_bibtex_files(bib_files)
    sys.stderr = sstderr
    print("[CITATIONS PARSED]", ID)
    arxiv = extract_arxiv_from_entries(entries, citation_keys)
    clear_directory(ID)
    print("[DONE]", ID)


    return arxiv


def scrape(ID):
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()
    with engine.connect() as session:
        stmt = select(Citations).where(Citations.c.origin_publication_id == ID).limit(1)
        res = session.execute(stmt)
        if len(list(res)) > 0:
            return []
    
    internet_part(ID)
    return cpu_part(ID)


def main():
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()

    MilvusSetterDB.create_collectio_metas()
    MilvusSetterDB.create_collection_papers()
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )

    
        
    pdb = get_papers_db(flag='r')
    keys = list(pdb.keys())

    for key in keys:
        result = scrape(key)
        for citation in result:
            with engine.connect() as session:
                stmt = insert(Citations).values(origin_publication_id=key, citation_publication_id=citation)
                session.execute(stmt)
                session.commit()
    


        
    

    
        

    

if __name__ == '__main__':
    main()
