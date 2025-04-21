from aslite.db import *
def convert_bool_list_to_bytes(bool_list):
    if len(bool_list) % 8 != 0:
        raise ValueError("The length of a boolean list must be a multiple of 8")

    byte_array = bytearray(len(bool_list) // 8)
    for i, bit in enumerate(bool_list):
        if bit == 1:
            index = i // 8
            shift = i % 8
            byte_array[index] |= (1 << shift)
    return bytes(byte_array)


def test_chemical_embeddings_db(): # temporary test code, should not be run on a full db
    import random
    embedding_db: MilvusClient = get_embeddings_db()
    print(embedding_db.describe_collection("chemical_embeddings"))
    random_embeddings = [[bool(random.randint(0, 1)) for _ in range(config.chemical_embedding_size)] for _ in range(1000)]
    data = [
    {
        "chemical_embedding": convert_bool_list_to_bytes(random_embeddings[i]),
        "tags": ["test", "test 1" if random.randint(0, 1)==0 else "test2"],
        "category": "chemistry?",
        "paper_id": str(random.randint(1, 1000)),
        "SMILES": "CN=C=O"
    } for i in range(len(random_embeddings))
    ]
    res = embedding_db.insert(collection_name="chemical_embeddings", data=data)
    print(res)
    random_vector_q = [bool(random.randint(0, 1)) for _ in range(config.chemical_embedding_size)]
    print(random_vector_q)
    res = embedding_db.search(
        collection_name="chemical_embeddings",
        data=[convert_bool_list_to_bytes(random_vector_q)],
        limit=5,
        anns_field="chemical_embedding",
        filter='ARRAY_CONTAINS(tags, "test")',
        search_params={"metric_type": "JACCARD"},
    )
    print(res)
    res = embedding_db.search(
        collection_name="chemical_embeddings",
        data=[convert_bool_list_to_bytes(random_vector_q)],
        limit=5,
        anns_field="chemical_embedding",
        filter='ARRAY_CONTAINS(tags, "test1")',
        search_params={"metric_type": "JACCARD"},
    )
    print(res)
    res = embedding_db.search(
        collection_name="chemical_embeddings",
        data=[convert_bool_list_to_bytes(random_vector_q)],
        limit=5,
        anns_field="chemical_embedding",
        filter='ARRAY_CONTAINS(tags, "test2")',
        search_params={"metric_type": "JACCARD"},
    )
    print(res)

test_chemical_embeddings_db()
