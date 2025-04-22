"""
Flask server backend

ideas:
- allow delete of tags
- unify all different pages into single search filter sort interface
- special single-image search just for paper similarity
"""

import os
import re
import time
from random import shuffle

import numpy as np
from sklearn import svm

from image_search import hybrid_search, get_image_path, FigureVectorizer

vectorizer = FigureVectorizer("cuda" if torch.cuda.is_available() else "cpu")

from flask import g  # global session-level object
from flask import Flask, redirect, render_template, request, session, url_for
from PIL import Image

from aslite.db import (
    get_email_db,
    get_last_active_db,
    get_metas_db,
    get_images_db,
    get_papers_db,
    get_embeddings_db,
    get_tags_db,
    load_features,
)

from db.SQLLiteAlchemyInstance import SQLAlchemyInstance
from db.SQLLite.OrmDB import Users, SeenPublications, Publications
from papers.paper import Paper
from sqlalchemy import insert, select, func

from aslite.db import get_papers_db, get_metas_db, get_tags_db, get_last_active_db, get_email_db
from aslite.db import load_features
from algorithms.rl_algorithm.rl_algorithm import RLAlgorithm
from papers.paper import Paper
from rdkit import Chem
from aslite.fingerprint import calculate_embedding


# -----------------------------------------------------------------------------
# inits and globals

RET_NUM = 25 # number of papers to return per page

app = Flask(__name__)

# set the secret key so we can cryptographically sign cookies and maintain sessions
if os.path.isfile('secret_key.txt'):
    # example of generating a good key on your system is:
    # import secrets; secrets.token_urlsafe(16)
    sk = open('secret_key.txt').read().strip()
else:
    print("WARNING: no secret key found, using default devkey")
    sk = 'devkey'
app.secret_key = sk

# -----------------------------------------------------------------------------
# globals that manage the (lazy) loading of various state for a request

def get_tags():
    if g.user is None:
        return {}
    if not hasattr(g, '_tags'):
        with get_tags_db() as tags_db:
            tags_dict = tags_db[g.user] if g.user in tags_db else {}
        g._tags = tags_dict
    return g._tags

def get_papers():
    if not hasattr(g, '_pdb'):
        g._pdb = get_papers_db()
    return g._pdb

def get_metas():
    if not hasattr(g, '_mdb'):
        g._mdb = get_metas_db()
    return g._mdb


def get_images():
    if not hasattr(g, "_idb"):
        g._idb = get_images_db()
    return g._idb


def get_embeddings():
    if not hasattr(g, "_edb"):
        g._edb = get_embeddings_db()
    return g._edb


@app.before_request
def before_request():
    g.user = session.get('user', None)

    # record activity on this user so we can reserve periodic
    # recommendations heavy compute only for active users
    if g.user:
        with get_last_active_db(flag='c') as last_active_db:
            last_active_db[g.user] = int(time.time())

@app.teardown_request
def close_connection(error=None):
    # close any opened database connections
    if hasattr(g, '_pdb'):
        g._pdb.close()
    if hasattr(g, '_mdb'):
        g._mdb.close()

# -----------------------------------------------------------------------------
# ranking utilities for completing the search/rank/filter requests

def render_pid(pid):
    # render a single paper with just the information we need for the UI
    pdb = get_papers()
    tags = get_tags()
    thumb_path = 'static/thumb/' + pid + '.jpg'
    thumb_url = thumb_path if os.path.isfile(thumb_path) else ''
    d = pdb[pid]
    return dict(
        weight = 0.0,
        id = d['_id'],
        url = d['url'],
        title = d['title'],
        time = d['_time_str'],
        authors = ', '.join(a['name'] for a in d['authors']),
        tags = ', '.join(t['term'] for t in d['tags']),
        utags = [t for t, pids in tags.items() if pid in pids],
        summary = d['summary'],
        thumb_url = thumb_url,
    )


def render_iid(iid):
    # render a single image with just the information we need for the UI
    idb = get_images()
    d = idb[iid]
    path = get_image_path(iid)
    url = path if os.path.isfile(path) else ''

    arxiv_id = d["base_id"]
    if d["version"] > 0:
        arxiv_id += "v" + d["version"]

    return dict(weight=0.0, id=arxiv_id, path=url, caption=d["caption"])


def random_rank():
    mdb = get_metas()
    pids = list(mdb.keys())
    shuffle(pids)
    scores = [0 for _ in pids]
    return pids, scores

def time_rank():
    mdb = get_metas()
    ms = sorted(mdb.items(), key=lambda kv: kv[1]["_time"], reverse=True)
    tnow = time.time()
    pids = [k for k, v in ms]
    scores = [(tnow - v["_time"]) / 60 / 60 / 24 for k, v in ms]  # time delta in days
    return pids, scores



def svm_rank(tags: str = "", pid: str = "", C: float = 0.01):
    # tag can be one tag or a few comma-separated tags or 'all' for all tags we have in db
    # pid can be a specific paper id to set as positive for a kind of nearest neighbor search
    if not (tags or pid):
        return [], [], []

    # load all of the features
    features = load_features()
    x, pids = features["x"], features["pids"]
    n, d = x.shape
    ptoi, itop = {}, {}
    for i, p in enumerate(pids):
        ptoi[p] = i
        itop[i] = p

    # construct the positive set
    y = np.zeros(n, dtype=np.float32)
    if pid:
        y[ptoi[pid]] = 1.0
    elif tags:
        tags_db = get_tags()
        tags_filter_to = tags_db.keys() if tags == "all" else set(tags.split(","))
        for tag, pids in tags_db.items():
            if tag in tags_filter_to:
                for pid in pids:
                    y[ptoi[pid]] = 1.0

    if y.sum() == 0:
        return [], [], []  # there are no positives?

    # classify
    clf = svm.LinearSVC(
        class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=C
    )
    clf.fit(x, y)
    s = clf.decision_function(x)
    sortix = np.argsort(-s)
    pids = [itop[ix] for ix in sortix]
    scores = [100 * float(s[ix]) for ix in sortix]

    # get the words that score most positively and most negatively for the svm
    ivocab = {v: k for k, v in features["vocab"].items()}  # index to word mapping
    weights = clf.coef_[0]  # (n_features,) weights of the trained svm
    sortix = np.argsort(-weights)
    words = []
    for ix in list(sortix[:40]) + list(sortix[-20:]):
        words.append(
            {
                "word": ivocab[ix],
                "weight": weights[ix],
            }
        )

    return pids, scores, words


def search_rank(q: str = ""):
    if not q:
        return [], []  # no query? no results
    qs = q.lower().strip().split()  # split query by spaces and lowercase

    pdb = get_papers()
    match = lambda s: sum(min(3, s.lower().count(qp)) for qp in qs)
    matchu = lambda s: sum(int(s.lower().count(qp) > 0) for qp in qs)
    pairs = []
    for pid, p in pdb.items():
        score = 0.0
        score += 10.0 * matchu(" ".join([a["name"] for a in p["authors"]]))
        score += 20.0 * matchu(p["title"])
        score += 1.0 * match(p["summary"])
        if score > 0:
            pairs.append((score, pid))

    pairs.sort(reverse=True)
    pids = [p[1] for p in pairs]
    scores = [p[0] for p in pairs]
    return pids, scores

def chemical_formulas_rank(input_SMILES: str = '', limit: int = 100):
    client = get_embeddings_db()

    # Convert the input SMILES into a fingerprint and then into a binary vector for Milvus.
    fp = calculate_embedding(input_SMILES)
    if fp is None:
        raise ValueError("Invalid input SMILES string.")

    # Run a Milvus search query using the binary vector.
    # Adjust the search "limit" as needed to capture enough SMILES entries for each paper.
    search_results = client.search(
        collection_name="chemical_embeddings",
        data=[fp],
        limit=limit,
        anns_field="chemical_embedding",
        output_fields=["paper_id", "SMILES"],
        filter="",  # If you want to restrict the search, add your filter here.
        search_params={"metric_type": "JACCARD"}
    )

    # Milvus returns a list (one per query vector) of hits. Each hit typically includes the field values and a score.
    # Note: When using JACCARD in Milvus, the score is a distance so lower values mean better matches.
    # We convert this to a similarity by computing: similarity = 1 - distance.
    paper_scores = {}
    # Iterate over the hits for our single query.
    for hit in search_results[0]:
        # Assume each hit.entity is a dictionary containing "paper_id" and "SMILES".
        print(hit)
        paper_id = str(hit['entity']['paper_id'])
        # Convert distance to similarity.
        similarity = 1 - hit['distance']
        # For each paper, keep the maximum similarity encountered.
        if paper_id not in paper_scores or paper_scores[paper_id] < similarity:
            paper_scores[paper_id] = similarity

    # Sort the papers by similarity score (highest similarity first)
    sorted_results = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)
    pids, scores = zip(*sorted_results) if sorted_results else ([], [])
    print(pids)
    return list(pids), list(scores)



def image_rank(q: str, img: Image.Image):
    client = get_embeddings()

    if isinstance(img, np.ndarray) and q:
        text_emb, image_emb = vectorizer([img], [q])
        res = hybrid_search(client, image_emb, text_emb)
    elif q:
        text_emb = vectorizer.text_embedding([q]).cpu().tolist()
        res = client.search("images_collection", text_emb, anns_field="caption_embedding")
    else:
        image_emb = vectorizer.image_embedding([img]).cpu().tolist()
        res = client.search("images_collection", image_emb, anns_field="image_embedding")

    return [r["id"] for r in res[0]], [r["distance"] for r in res[0]]


# -----------------------------------------------------------------------------
# primary application endpoints

def default_context():
    # any global context across all pages, e.g. related to the current user
    context = {}
    context['user'] = g.user if g.user is not None else ''
    return context

def get_seen_pids_for_user():
    print('get_seen_pids_for_user', g.user)
    if g.user is None:
        return []

    instance = SQLAlchemyInstance()
    engine = instance.get_engine()

    with engine.connect() as conn:

        user_id = get_user()
        print(user_id)
        # SELECT arxiv_id from publications where there's a seen_publications record
        seen_stmt = (
            select(SeenPublications.origin_publication_id)
            .select_from(SeenPublications)
            .where(SeenPublications.user_id == user_id)
            .limit(50)
        )

        result = conn.execute(seen_stmt).fetchall()
        return result

def get_user():
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()
    if g.user is None:
        return []

    with engine.connect() as conn:
        if check_user_exists(g.user):
            user_stmt = select(Users.id).where(Users.name == g.user)
            user_row = conn.execute(user_stmt).fetchone()
            return user_row[0]


def add_user(name):
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()

    with engine.connect() as conn:
        if not check_user_exists(name):
            print('add_user', name)
            user_stmt = insert(Users).values(name=name)
            conn.execute(user_stmt)
            conn.commit()

def check_user_exists(name):
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()
    with engine.connect() as conn:
        user_stmt = select(Users.id).where(Users.name == name)
        user_row = conn.execute(user_stmt).fetchone()
        if not user_row:
            return False
        return True


def add_seen_publication(pid):
    instance = SQLAlchemyInstance()
    engine = instance.get_engine()
    if g.user is None:
        return []
    with engine.connect() as conn:
        user_id = get_user()

        # Check if exists
        check_stmt = (
            select(SeenPublications)
            .where(
                SeenPublications.user_id == user_id,
                SeenPublications.origin_publication_id == pid
            )
        )
        if conn.execute(check_stmt).first():
            print("Already exists, skipping.")
            return
        print("bu bu")
        # Insert manually
        insert_stmt = insert(SeenPublications).values(
            user_id=user_id,
            origin_publication_id=pid
        )
        conn.execute(insert_stmt)
        conn.commit()
        print("Added new seen publication.")


@app.route('/', methods=['GET'])
def main():

    # default settings
    default_rank = "time"
    default_tags = ""
    default_time_filter = ""
    default_skip_have = "no"

    # override variables with any provided options via the interface
    opt_rank = request.form.get("rank", default_rank)  # rank type. search|tags|pid|time|random
    opt_q = request.form.get("q", "")  # search request in the text box
    opt_tags = request.form.get("tags", default_tags)  # tags to rank by if opt_rank == 'tag'
    opt_pid = request.form.get("pid", "")  # pid to find nearest neighbors to
    opt_time_filter = request.form.get("time_filter", default_time_filter)  # number of days to filter by
    opt_skip_have = request.form.get("skip_have", default_skip_have)  # hide papers we already have?
    opt_svm_c = request.form.get("svm_c", "")  # svm C parameter
    opt_smiles_input = request.form.get("smiles_input", "")
    opt_page_number = request.form.get("page_number", "1")  # page number for pagination

    opt_image_input = request.files.get("image_input")

    # if a query is given, override rank to be of type "search"
    # this allows the user to simply hit ENTER in the search field and have the correct thing happen
    if opt_image_input:
        opt_image_input = Image.open(opt_image_input.stream)
        opt_rank = "image"
    elif opt_q and opt_rank != "image":
        opt_rank = "search"

    # try to parse opt_svm_c into something sensible (a float)
    try:
        C = float(opt_svm_c)
    except ValueError:
        C = 0.01 # sensible default, i think

    # crop the number of results to RET_NUM, and paginate
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1

    if opt_rank == "image":
        iids, scores = image_rank(opt_q, opt_image_input)

        # render all images to just the information we need for the UI
        images = [render_iid(iid) for iid in iids]
        for i, d in enumerate(images):
            d["weight"] = float(scores[i])

        context = default_context()
        context["images"] = images

    else:
        # rank papers: by tags, by time, by random
        words = []  # only populated in the case of svm rank
        if opt_rank == "search":
            pids, scores = search_rank(q=opt_q)
        elif opt_rank == "tags":
            pids, scores, words = svm_rank(tags=opt_tags, C=C)
        elif opt_rank == "pid":
            pids, scores, words = svm_rank(pid=opt_pid, C=C)
        elif opt_rank == "time":
            pids, scores = time_rank()
        elif opt_rank == "random":
            pids, scores = random_rank()
        elif opt_rank == "chemical_formulas":
            pids, scores = chemical_formulas_rank(opt_smiles_input)
        else:
            raise ValueError("opt_rank %s is not a thing" % (opt_rank,))
    # rank papers: by tags, by time, by random


    pdb = get_papers()

    pids = []
    words = []
    scores = []
    if g.user is not None and g.user != '':
        user_history = get_seen_pids_for_user()
        papers_history = []
        for i in range(0, len(user_history)):
            papers_history.append(Paper.from_id(user_history[i][0], pdb))

        recommend_instance = RLAlgorithm(get_papers_db())
        result_recommendations, scores = recommend_instance.recommend(papers_history, 20)
        for i in range(0, len(result_recommendations)):
            pids.append(result_recommendations[i].arxiv_id)


    else:
        words = []  # only populated in the case of svm rank
        if opt_rank == 'search':
            pids, scores = search_rank(q=opt_q)
        elif opt_rank == 'tags':
            pids, scores, words = svm_rank(tags=opt_tags, C=C)
        elif opt_rank == 'pid':
            pids, scores, words = svm_rank(pid=opt_pid, C=C)
        elif opt_rank == 'time':
            pids, scores = time_rank()
        elif opt_rank == 'random':
            pids, scores = random_rank()
        elif opt_rank == 'chemical_formulas':
            pids, scores = chemical_formulas_rank(opt_smiles_input)
        else:
            raise ValueError("opt_rank %s is not a thing" % (opt_rank,))

        # filter by time
        if opt_time_filter:
            mdb = get_metas()
            kv = {
                k: v for k, v in mdb.items()
            }  # read all of metas to memory at once, for efficiency
            tnow = time.time()
            deltat = (
                int(opt_time_filter) * 60 * 60 * 24
            )  # allowed time delta in seconds
            keep = [
                i for i, pid in enumerate(pids) if (tnow - kv[pid]["_time"]) < deltat
            ]
            pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]
        mdb = get_metas()
        kv = {
            k: v for k, v in mdb.items()
        }  # read all of metas to memory at once, for efficiency
        tnow = time.time()
        deltat = (
                int(opt_time_filter) * 60 * 60 * 24
        )  # allowed time delta in seconds
        keep = [
            i for i, pid in enumerate(pids) if (tnow - kv[pid]["_time"]) < deltat
        ]
        pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]

        # optionally hide papers we already have
        if opt_skip_have == "yes":
            tags = get_tags()
            have = set().union(*tags.values())
            keep = [i for i, pid in enumerate(pids) if pid not in have]
            pids, scores = [pids[i] for i in keep], [scores[i] for i in keep]
    # crop the number of results to RET_NUM, and paginate
    start_index = (page_number - 1) * RET_NUM  # desired starting index
    end_index = min(start_index + RET_NUM, len(pids))  # desired ending index
    pids = pids[start_index:end_index]
    scores = scores[start_index:end_index]
# crop the number of results to RET_NUM, and paginate
    try:
        page_number = max(1, int(opt_page_number))
    except ValueError:
        page_number = 1


    start_index = (page_number - 1) * RET_NUM # desired starting index
    end_index = min(start_index + RET_NUM, len(pids)) # desired ending index

    pids = pids[start_index:end_index]
    # scores = scores[start_index:end_index]

    # render all papers to just the information we need for the UI
    papers = [render_pid(pid) for pid in pids]
    for i, p in enumerate(papers):
        p['weight'] = float(scores[i])

        # build the current tags for the user, and append the special 'all' tag
        tags = get_tags()
        rtags = [{"name": t, "n": len(pids)} for t, pids in tags.items()]
        if rtags:
            rtags.append({"name": "all"})

        context = default_context()
        context["papers"] = papers
        context["words"] = words
        context["tags"] = rtags
        context["words_desc"] = (
            "Here are the top 40 most positive and bottom 20 most negative weights of the SVM. If they don't look great then try tuning the regularization strength hyperparameter of the SVM, svm_c, above. Lower C is higher regularization."
        )

    # build the page context information and render
    context = default_context()
    context['papers'] = papers
    context['tags'] = rtags
    context['words'] = words
    context['words_desc'] = "Here are the top 40 most positive and bottom 20 most negative weights of the SVM. If they don't look great then try tuning the regularization strength hyperparameter of the SVM, svm_c, above. Lower C is higher regularization."
    context['gvars'] = {}
    context['gvars']['rank'] = opt_rank
    context['gvars']['tags'] = opt_tags
    context['gvars']['pid'] = opt_pid
    context['gvars']['time_filter'] = opt_time_filter
    context['gvars']['skip_have'] = opt_skip_have
    context['gvars']['search_query'] = opt_q
    context['gvars']['svm_c'] = str(C)
    context['gvars']['page_number'] = str(page_number)
    return render_template('index.html', **context)

@app.route('/inspect', methods=['GET'])
def inspect():

    # fetch the paper of interest based on the pid
    pid = request.args.get('pid', '')
    pdb = get_papers()
    recommend_instance = RLAlgorithm(get_papers_db())
    if pid not in pdb:
        return "error, malformed pid" # todo: better error handling

    # load the tfidf vectors, the vocab, and the idf table
    features = load_features()
    x = features['x']
    idf = features['idf']
    ivocab = {v:k for k,v in features['vocab'].items()}
    pix = features['pids'].index(pid)
    wixs = np.flatnonzero(np.asarray(x[pix].todense()))
    words = []
    for ix in wixs:
        words.append({
            'word': ivocab[ix],
            'weight': float(x[pix, ix]),
            'idf': float(idf[ix]),
        })
    words.sort(key=lambda w: w['weight'], reverse=True)

    # package everything up and render
    paper = render_pid(pid)
    add_seen_publication(pid)

    paper_instance = Paper.from_id(pid, pdb)
    result_recommendations = recommend_instance.recommend([paper_instance], 5)
    similar_papers = [render_pid(p.arxiv_id) for p in result_recommendations if p.arxiv_id != pid]

    # print(result_recommendations)
    context = default_context()
    context['paper'] = paper
    context['words'] = words
    context['words_desc'] = "The following are the tokens and their (tfidf) weight in the paper vector. This is the actual summary that feeds into the SVM to power recommendations, so hopefully it is good and representative!"
    context['similar_papers'] = similar_papers
    return render_template('inspect.html', **context)

@app.route('/add_to_folder/<folder>/<pid>')
def add_to_folder(folder, pid):
    if g.user is None:
        return "error, not logged in", 401

    with get_tags_db(flag='c') as tags_db:
        if g.user not in tags_db:
            tags_db[g.user] = {}

        user_tags = tags_db[g.user]
        user_tags.setdefault(folder, set()).add(pid)
        tags_db[g.user] = user_tags

    return redirect(url_for('profile'))


@app.route('/remove_from_folder/<folder>/<pid>')
def remove_from_folder(folder, pid):
    if g.user is None:
        return "error, not logged in", 401

    with get_tags_db(flag='c') as tags_db:
        user_tags = tags_db.get(g.user, {})
        if folder in user_tags and pid in user_tags[folder]:
            user_tags[folder].remove(pid)
            if not user_tags[folder]:
                del user_tags[folder]
            tags_db[g.user] = user_tags

    return redirect(url_for('profile'))

@app.route('/stats')
def stats():
    context = default_context()
    mdb = get_metas()
    kv = {k:v for k,v in mdb.items()} # read all of metas to memory at once, for efficiency
    times = [v['_time'] for v in kv.values()]
    tstr = lambda t: time.strftime('%b %d %Y', time.localtime(t))

    context['num_papers'] = len(kv)
    if len(kv) > 0:
        context['earliest_paper'] = tstr(min(times))
        context['latest_paper'] = tstr(max(times))
    else:
        context['earliest_paper'] = 'N/A'
        context['latest_paper'] = 'N/A'

    # count number of papers from various time deltas to now
    tnow = time.time()
    for thr in [1, 6, 12, 24, 48, 72, 96]:
        context['thr_%d' % thr] = len([t for t in times if t > tnow - thr*60*60])

    return render_template('stats.html', **context)

@app.route('/about')
def about():
    context = default_context()
    return render_template('about.html', **context)




@app.route('/settings')
def settings():
    context = default_context()
    from flask import request
    with get_email_db() as edb:
        email = edb.get(g.user, '')
    context['email'] = email
    context['notif_enabled'] = bool(email)

    context['frequency'] = request.args.get('frequency', 'daily')

    context['gvars'] = {
        'skip_have': request.args.get('skip_have', 'no'),
        'rank':      request.args.get('rank', 'time'),
    }

    return render_template('settings.html', **context)

@app.route('/profile/bookmarks/<folder>')
def view_folder(folder):
    context = default_context()
    tags = get_tags()
    print(tags)
    if folder not in tags:
        return "folder not found", 404

    pids = list(tags[folder])
    papers = [ render_pid(pid) for pid in pids ]
    context.update({
        'current_folder': folder,
        'papers': papers
    })
    return render_template('folder.html', **context)

@app.route('/profile')
def profile():
    context = default_context()
    with get_email_db() as edb:
        email = edb.get(g.user, '')
        context['email'] = email
    return render_template('profile.html', **context)


# -----------------------------------------------------------------------------
# tag related endpoints: add, delete tags for any paper

@app.route('/add/<pid>/<tag>')
def add(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"
    if tag == 'all':
        return "error, cannot add the protected tag 'all'"
    elif tag == 'null':
        return "error, cannot add the protected tag 'null'"

    with get_tags_db(flag='c') as tags_db:

        # create the user if we don't know about them yet with an empty library
        if not g.user in tags_db:
            tags_db[g.user] = {}

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            d[tag] = set()
        d[tag].add(pid)

        # write back to database
        tags_db[g.user] = d

    print("added paper %s to tag %s for user %s" % (pid, tag, g.user))
    return "ok: " + str(d) # return back the user library for debugging atm

@app.route('/sub/<pid>/<tag>')
def sub(pid=None, tag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag='c') as tags_db:

        # if the user doesn't have any tags, there is nothing to do
        if not g.user in tags_db:
            return "user has no library of tags ¯\_(ツ)_/¯"

        # fetch the user library object
        d = tags_db[g.user]

        # add the paper to the tag
        if tag not in d:
            return "user doesn't have the tag %s" % (tag, )
        else:
            if pid in d[tag]:

                # remove this pid from the tag
                d[tag].remove(pid)

                # if this was the last paper in this tag, also delete the tag
                if len(d[tag]) == 0:
                    del d[tag]

                # write back the resulting dict to database
                tags_db[g.user] = d
                return "ok removed pid %s from tag %s" % (pid, tag)
            else:
                return "user doesn't have paper %s in tag %s" % (pid, tag)

@app.route('/del/<tag>')
def delete_tag(tag=None):
    if g.user is None:
        return "error, not logged in"

    with get_tags_db(flag='c') as tags_db:

        if g.user not in tags_db:
            return "user does not have a library"

        d = tags_db[g.user]

        if tag not in d:
            return "user does not have this tag"

        # delete the tag
        del d[tag]

        # write back to database
        tags_db[g.user] = d

    print("deleted tag %s for user %s" % (tag, g.user))
    return "ok: " + str(d) # return back the user library for debugging atm

# -----------------------------------------------------------------------------
# endpoints to log in and out

@app.route('/login', methods=['POST'])
def login():

    # the user is logged out but wants to log in, ok
    if g.user is None and request.form['username']:
        username = request.form['username']
        if len(username) > 0: # one more paranoid check
            add_user(username)
            session['user'] = username

    return redirect(url_for('profile'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('profile'))

# -----------------------------------------------------------------------------
# user settings and configurations

@app.route('/register_email', methods=['POST'])
def register_email():
    email = request.form['email']

    if g.user:
        # do some basic input validation
        proper_email = re.match(r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}$', email, re.IGNORECASE)
        if email == '' or proper_email: # allow empty email, meaning no email
            # everything checks out, write to the database
            with get_email_db(flag='c') as edb:
                edb[g.user] = email

    return redirect(url_for('profile'))
