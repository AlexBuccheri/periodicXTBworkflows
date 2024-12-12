import os
from sys import prefix 

# Define portable, absolute path to crystal CIF files
project_name = 'tb_benchmarking'
relative_cif_root = 'data/bulk_crystals/cifs/'

dir_path = os.path.dirname(os.path.realpath(__file__))
prefix = dir_path.split(project_name)[0]

cif_root = os.path.join(prefix, project_name, relative_cif_root)
