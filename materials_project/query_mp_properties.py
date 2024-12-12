"""Functions for querying Materials Project API

Example:

from typing import Dict
import os
import pickle

from emmet.core.mpid import MPID
from pymatgen.core.structure import Structure

if __name__ == "__main__":
    api_key = os.getenv('MAPI_KEY')
    material_ids = ["mp-149"]
    docs = query_properties(api_key, material_ids=[MPID(id) for id in material_ids], fields=["structure"])

    structures: Dict[str, Structure] = {material_ids[i]:doc.structure for i, doc in enumerate(docs)}

    # Serialize (pickle) the Structure object to a file
    with open("materials_project/structures/structures.pkl", "wb") as file:
        pickle.dump(structures, file)
"""
from mp_api.client import MPRester


def query_properties(api_key, **kwargs):
    """Query materials project entries.

     print(docs[i]) will show all attributes
    """
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(**kwargs)
    return docs
