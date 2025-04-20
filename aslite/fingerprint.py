from skfp.fingerprints import RDKitFingerprint
from aslite import config
import numpy as np

# Helper function to convert a list of booleans (or 0/1 integers) to bytes.
# (Assumes that such a conversion is needed by your Milvus client.)
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

def calculate_embedding(smiles: str) -> np.ndarray | None:
    try:
        emb = RDKitFingerprint(fp_size=config.chemical_embedding_size).transform([smiles])[0]
    except ValueError:
        return None
    if emb is None:
        return None
    return convert_bool_list_to_bytes(emb)
