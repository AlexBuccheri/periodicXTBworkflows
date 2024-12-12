""" Containers for crystal systems.
"""
from tb_lite.crystal_references import cubic, hexagonal, tetragonal, monoclinic
from tb_lite.src.parsers.parsers import cif_to_ase_atoms

# To Add:  GaP, InN, InAs, PbSe, graphene
# Bulk systems
bulk_materials = {'silicon': cubic.silicon(),
                  'germanium': cubic.germanium(),
                  'diamond': cubic.diamond(),
                  'zinc_oxide': cif_to_ase_atoms(hexagonal.hexagonal_cifs.get('zinc_oxide').file),
                  'mos2': cif_to_ase_atoms(hexagonal.hexagonal_cifs.get('molybdenum_disulfide').file),
                  'ws2': cif_to_ase_atoms(hexagonal.hexagonal_cifs.get('tungsten_disulfide').file),
                  'bn_hex': hexagonal.boron_nitride_hexagonal(),
                  'bn_cubic': cif_to_ase_atoms(cubic.fcc_cifs.get('boron_nitride').file),
                  'mgo': cif_to_ase_atoms(cubic.fcc_cifs.get('magnesium_oxide').file),
                  'copper': cif_to_ase_atoms(cubic.fcc_cifs.get('copper').file),
                  'nacl': cif_to_ase_atoms(cubic.fcc_cifs.get('sodium_chloride').file),
                  'zro2': cif_to_ase_atoms(cubic.fcc_cifs.get('zirconium_dioxide').file),
                  'pbs': cif_to_ase_atoms(cubic.fcc_cifs.get('lead_sulfide').file),
                  'tio2_rutile': tetragonal.tio2_rutile(),
                  'tio2_ana': tetragonal.tio2_anatase(),
                  'cdse': cif_to_ase_atoms(hexagonal.hexagonal_cifs.get('cadmium_selenide').file),
                  'gan': cif_to_ase_atoms(cubic.fcc_cifs.get('gallium_nitride').file),
                  'graphite': cif_to_ase_atoms(hexagonal.hexagonal_cifs.get('graphite').file),
                  'gaas': cif_to_ase_atoms(cubic.fcc_cifs.get('gallium_arsenide').file),
                  'wo3_monoclinic': cif_to_ase_atoms(monoclinic.simple_monoclinic_cifs.get("tungsten_oxide").file),
                  'pbte': cif_to_ase_atoms(cubic.fcc_cifs.get("lead_telluride").file)
                  #'inp':        cif_to_ase_atoms(cubic.fcc_cifs.get("indium_phosphide").file), # Caused a crash
                }
