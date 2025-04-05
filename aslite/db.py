"""
Database support functions.
The idea is that none of the individual scripts deal directly with the file system.
Any of the file system I/O and the associated settings are in this single file.
"""

import os
import sqlite3, zlib, pickle, tempfile
from sqlitedict import SqliteDict
from contextlib import contextmanager
from pymilvus import MilvusClient, DataType
from aslite import config

# -----------------------------------------------------------------------------
# global configuration

DATA_DIR = 'data'

# -----------------------------------------------------------------------------
# utilities for safe writing of a pickle file

# Context managers for atomic writes courtesy of
# http://stackoverflow.com/questions/2333872/atomic-writing-to-file-with-python
@contextmanager
def _tempfile(*args, **kws):
    """ Context for temporary file.
    Will find a free temporary filename upon entering
    and will try to delete the file on leaving
    Parameters
    ----------
    suffix : string
        optional file suffix
    """

    fd, name = tempfile.mkstemp(*args, **kws)
    os.close(fd)
    try:
        yield name
    finally:
        try:
            os.remove(name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise e


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.
    Allows reading and writing to and from the same filename.
    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.pop('fsync', False)

    with _tempfile(dir=os.path.dirname(filepath)) as tmppath:
        with open(tmppath, *args, **kwargs) as f:
            yield f
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.rename(tmppath, filepath)

def safe_pickle_dump(obj, fname):
    """
    prevents a case where one process could be writing a pickle file
    while another process is reading it, causing a crash. the solution
    is to write the pickle file to a temporary file and then move it.
    """
    with open_atomic(fname, 'wb') as f:
        pickle.dump(obj, f, -1) # -1 specifies highest binary protocol

# -----------------------------------------------------------------------------

class CompressedSqliteDict(SqliteDict):
    """ overrides the encode/decode methods to use zlib, so we get compressed storage """

    def __init__(self, *args, **kwargs):

        def encode(obj):
            return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

        def decode(obj):
            return pickle.loads(zlib.decompress(bytes(obj)))

        super().__init__(*args, **kwargs, encode=encode, decode=decode)

# -----------------------------------------------------------------------------
"""
some docs to self:
flag='c': default mode, open for read/write, and creating the db/table if necessary
flag='r': open for read-only
"""

MILVUS_MODE = os.getenv("MILVUS_MODE", "lite")

# stores info about papers, and also their lighter-weight metadata
PAPERS_DB_FILE = os.path.join(DATA_DIR, 'papers.db')
# stores account-relevant info, like which tags exist for which papers
DICT_DB_FILE = os.path.join(DATA_DIR, 'dict.db')
EMBEDDING_DB_FILE = "http://localhost:19530/" if MILVUS_MODE == "standalone" else os.path.join(DATA_DIR, 'embeddings.db')
def get_papers_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    pdb = CompressedSqliteDict(PAPERS_DB_FILE, tablename='papers', flag=flag, autocommit=autocommit)
    return pdb

def get_metas_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    mdb = SqliteDict(PAPERS_DB_FILE, tablename='metas', flag=flag, autocommit=autocommit)
    return mdb

def get_tags_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    tdb = CompressedSqliteDict(DICT_DB_FILE, tablename='tags', flag=flag, autocommit=autocommit)
    return tdb

def get_last_active_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    ladb = SqliteDict(DICT_DB_FILE, tablename='last_active', flag=flag, autocommit=autocommit)
    return ladb

def get_email_db(flag='r', autocommit=True):
    assert flag in ['r', 'c']
    edb = SqliteDict(DICT_DB_FILE, tablename='email', flag=flag, autocommit=autocommit)
    return edb

def setup_chemical_embeddings_collection(client: MilvusClient):    
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="chemical_embedding", datatype=DataType.BINARY_VECTOR, dim=config.chemical_embedding_size)
    schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=127)
    schema.add_field(field_name="SMILES", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="tags",datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=127)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="chemical_embedding", index_type=config.chemical_index_type, metric_type="JACCARD") 
    client.create_collection(
        collection_name="chemical_embeddings",
        schema=schema,
        index_params=index_params,
        consistency_level=config.consistency_level
    )

def get_embeddings_db():
    client = MilvusClient(EMBEDDING_DB_FILE)
    if not client.has_collection("chemical_embeddings"):
        setup_chemical_embeddings_collection(client)
    return client


class EmbeddingsDB:
    def __init__(self) -> MilvusClient:
        self.client = get_embeddings_db()

    def __enter__(self) -> MilvusClient:
        return self.client

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.client.close()


# -----------------------------------------------------------------------------
"""
our "feature store" is currently just a pickle file, may want to consider hdf5 in the future
"""

# stores tfidf features a bunch of other metadata
FEATURES_FILE = os.path.join(DATA_DIR, 'features.p')

def save_features(features):
    """ takes the features dict and save it to disk in a simple pickle file """
    safe_pickle_dump(features, FEATURES_FILE)

def load_features():
    """ loads the features dict from disk """
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)
    return features
