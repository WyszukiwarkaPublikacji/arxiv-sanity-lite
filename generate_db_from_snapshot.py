import argparse
import json
import logging
import time

from tqdm import tqdm

from aslite.db import get_metas_db, get_papers_db

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    parser = argparse.ArgumentParser(description="Arxiv DB generator")
    parser.add_argument("-f", "--file", type=str, help="The file to generate db from")
    args = parser.parse_args()
    print(args)
    pdb = get_papers_db(flag="c")
    mdb = get_metas_db(flag="c")
    prevn = len(pdb)

    def store(p):
        pdb[p["_id"]] = p
        mdb[p["_id"]] = {"_time": p["_time"]}

    f = open(args.file, "r")
    flen = 0
    for _ in f:
        flen += 1
    f.seek(0)
    j = 0
    for line in tqdm(f, total=flen):
        js = json.loads(line)
        enc = {}
        enc["_idv"] = js["id"] + js["versions"][-1]["version"]
        enc["_id"] = js["id"]
        enc["_version"] = js["versions"][-1]["version"]
        enc["id"] = "http://arxiv.org/abs/" + enc["_idv"]
        enc["_time"] = time.mktime(time.strptime(js["update_date"], "%Y-%M-%d"))
        enc["_time_str"] = time.strftime(
            "%b %d %Y", time.strptime(js["update_date"], "%Y-%M-%d")
        )
        enc["summary"] = js["abstract"]
        enc["title"] = js["title"]
        enc["authors"] = []
        for i in js["authors"].split(","):
            enc["authors"].append({"name": i})
        enc["arxiv-comment"] = js["comments"]
        enc["arxiv_primary_category"] = js["categories"].split()[0]
        enc["tags"] = [{"term": i} for i in js["categories"].split(" ")]
        pdb[enc["_id"]] = enc
        mdb[enc["_id"]] = {"_time": enc["_time"]}
