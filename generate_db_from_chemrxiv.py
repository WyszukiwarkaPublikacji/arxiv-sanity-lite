# The file to generate the db from is avabile at: https://github.com/chemrxiv-dashboard/chemrxiv-dashboard.github.io/raw/refs/heads/master/data/allchemrxiv_data.json.bz2

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
    parser.add_argument("-f", "--file", type=str, required=True, help="The file to generate db from")
    args = parser.parse_args()
    pdb = get_papers_db(flag="c")
    mdb = get_metas_db(flag="c")
    prevn = len(pdb)

    def store(p):
        pdb[p["_id"]] = p
        mdb[p["_id"]] = {"_time": p["_time"]}

    with open(args.file, "r") as f:
        jsn = json.load(f)
    for paperid in tqdm(jsn):
        paper = jsn[paperid]
        if not paper["isLatestVersion"]:
            continue
        enc = {}
        enc["_idv"] = paper["id"] + "v" + paper["version"]
        enc["_id"] = paper["id"]
        enc["_version"] = paper["version"]
        enc["url"] = (
            "https://chemrxiv.org/engage/chemrxiv/article-details/" + paper["id"]
        )
        enc["_time"] = time.mktime(time.strptime(paper["statusDate"][:10], "%Y-%M-%d"))
        enc["_time_str"] = time.strftime(
            "%b %d %Y", time.strptime(paper["statusDate"][:10], "%Y-%M-%d")
        )
        enc["summary"] = paper["abstract"]
        enc["title"] = paper["title"]
        enc["authors"] = []
        for i in paper["authors"]:
            enc["authors"].append({"name": i["firstName"] + i["lastName"]})
        enc["arxiv-comment"] = "A chemrxiv publication"
        enc["arxiv_primary_category"] = paper["categories"][0]["name"]
        enc["tags"] = [{"term": i["name"]} for i in paper["categories"]]
        enc["doi"] = paper["doi"]
        enc["provider"] = "chemrxiv"
        pdb[enc["_id"]] = enc
        mdb[enc["_id"]] = {"_time": enc["_time"]}
