from skfp.fingerprints import RDKitFingerprint
from aslite import config
import numpy as np
def calculate_embedding(smiles: str) -> np.ndarray:
    return RDKitFingerprint(fp_size=config.chemical_embedding_size).transform([smiles])[0]
