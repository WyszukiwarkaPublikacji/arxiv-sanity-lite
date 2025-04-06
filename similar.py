from aslite.db import get_papers_db, save_features
from random import choice
from requests import get

pdb = get_papers_db()
keys = pdb.keys()


def similar(id: str) -> str: 
    """
    should return the most similar paper to the gien one,
    currenty returns random paper
    """
    key = choice(list(keys))
    # return pdb[key]["id"] #arxiv link for test purposes
    return pdb[key]["_id"]


