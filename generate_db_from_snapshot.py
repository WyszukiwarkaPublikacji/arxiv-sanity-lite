import argparse
import json
import logging
import os
import sys
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
    parser.add_argument("-f", "--file", type=str, help="The file to generate db from. If not specified, '$DOWNLOADS_DIR/data/downloads/<arxiv|chemrxiv>.json' is used, where the DOWNLOADS_DIR variable defaults to current working dir.")
    parser.add_argument("-a", "--arxiv", help="The file given is from an arxiv snapshot.", action="store_true")
    parser.add_argument("-c", "--chemrxiv", help="The file given is from a chemrxiv snapshot.", action="store_true")
    args = parser.parse_args()

    downloads_dir = os.environ.get("DOWNLOADS_DIR", os.path.join(os.getcwd(), "data/downloads"))

    pdb = get_papers_db(flag="c")
    mdb = get_metas_db(flag="c")
    if args.arxiv and args.chemrxiv:
        logging.error("specify either chemrxiv or arxiv")
        sys.exit(1)
    if not (args.arxiv or args.chemrxiv):
        logging.error("specify source format, see help for more information.")
        sys.exit(1)
    if args.arxiv:
        filename = args.file if args.file is not None else os.path.join(downloads_dir, "arxiv.json")
        with open(filename, "r") as f:
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
                enc["url"] = "http://arxiv.org/abs/" + enc["_idv"]
                enc["_time"] = time.mktime(time.strptime(js["update_date"], "%Y-%M-%d"))
                enc["_time_str"] = time.strftime("%b %d %Y", time.strptime(js["update_date"], "%Y-%M-%d"))
                enc["summary"] = js["abstract"]
                enc["title"] = js["title"]
                enc["authors"] = []
                for i in js["authors"].split(","):
                    enc["authors"].append({"name": i})
                enc["arxiv-comment"] = js["comments"]
                enc["arxiv_primary_category"] = js["categories"].split()[0]
                enc["tags"] = [{"term": i} for i in js["categories"].split(" ")]
                enc["doi"] = js["doi"]
                enc["provider"] = "arxiv"
                enc["computed_chemical"] = False
                pdb[enc["_id"]] = enc
                mdb[enc["_id"]] = {"_time": enc["_time"]}
    if args.chemrxiv:
        filename = args.file if args.file is not None else os.path.join(downloads_dir, "chemrxiv.json")
        with open(filename, "r") as f:
            jsn = json.load(f)
        for paperid in tqdm(jsn):
            paper = jsn[paperid]
            if not paper["isLatestVersion"]:
                continue
            enc = {}
            enc["_idv"] = paper["id"] + "v" + paper["version"]
            enc["_id"] = paper["id"]
            enc["_version"] = paper["version"]
            enc["url"] = "https://chemrxiv.org/engage/chemrxiv/article-details/" + paper["id"]
            enc["_time"] = time.mktime(time.strptime(paper["statusDate"][:10], "%Y-%M-%d"))
            enc["_time_str"] = time.strftime("%b %d %Y", time.strptime(paper["statusDate"][:10], "%Y-%M-%d"))
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
            enc["computed_chemical"] = False
            pdb[enc["_id"]] = enc
            mdb[enc["_id"]] = {"_time": enc["_time"]}
