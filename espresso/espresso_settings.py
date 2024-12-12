"""

"""
import os
from pathlib import Path
from typing import Optional

# Rutgers Pseudos , with keys equal to those in bulk_materials
rutgers_pseudos = {'silicon': {'Si': 'si_pbesol_v1.uspp.F.UPF'},
                   'germanium': {'Ge': 'ge_pbesol_v1.4.uspp.F.UPF'},
                   'diamond': {'C': 'c_pbesol_v1.2.uspp.F.UPF'},
                   'zinc_oxide': {'Zn': "zn_pbesol_v1.uspp.F.UPF", 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'mos2': {'Mo': 'mo_pbesol_v1.uspp.F.UPF', "S": 's_pbesol_v1.4.uspp.F.UPF'},
                   'ws2': {'W': 'w_pbesol_v1.2.uspp.F.UPF', "S": 's_pbesol_v1.4.uspp.F.UPF'},
                   'bn_hex': {'B': 'b_pbesol_v1.4.uspp.F.UPF', 'N': 'n_pbesol_v1.2.uspp.F.UPF'},
                   'bn_cubic': {'B': 'b_pbesol_v1.4.uspp.F.UPF', 'N': 'n_pbesol_v1.2.uspp.F.UPF'},
                   'mgo': {'Mg': 'mg_pbesol_v1.4.uspp.F.UPF', 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'copper': {'Cu': 'cu_pbesol_v1.2.uspp.F.UPF'},
                   'nacl': {'Na': 'na_pbesol_v1.5.uspp.F.UPF', 'Cl': 'cl_pbesol_v1.4.uspp.F.UPF'},
                   'zro2': {'Zr': 'zr_pbesol_v1.uspp.F.UPF', 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'pbs': {'Pb': 'pb_pbesol_v1.uspp.F.UPF', 'S': 's_pbesol_v1.4.uspp.F.UPF'},
                   'tio2_rutile': {'Ti': 'ti_pbesol_v1.4.uspp.F.UPF', 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'tio2_ana': {'Ti': 'ti_pbesol_v1.4.uspp.F.UPF', 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'cdse': {'Cd': 'cd_pbesol_v1.uspp.F.UPF', "Se": 'se_pbesol_v1.uspp.F.UPF'},
                   'gan': {'Ga': 'ga_pbesol_v1.4.uspp.F.UPF', 'N': 'n_pbesol_v1.2.uspp.F.UPF'},
                   'graphite': {'C': 'c_pbesol_v1.2.uspp.F.UPF'},
                   'gaas': {'Ga': 'ga_pbesol_v1.4.uspp.F.UPF', 'As': 'as_pbesol_v1.uspp.F.UPF'},
                   'wo3_monoclinic': {'W': 'w_pbesol_v1.2.uspp.F.UPF', 'O': 'o_pbesol_v1.2.uspp.F.UPF'},
                   'pbte': {'Pb': 'pt_pbesol_v1.4.uspp.F.UPF', 'Te': 'te_pbesol_v1.uspp.F.UPF'}
                   }


class EspressoSettings:
    env = {'ASE_ESPRESSO_COMMAND': "{binarypath}/pw.x -in PREFIX.pwi > PREFIX.pwo",
           'ESPRESSO_PSEUDO': ''}

    def __init__(self, binary_path: str, pseudo_dir: str, pseudos: Optional[dict] = None,
                 job_root='./', run_dir='./'):
        self.binary_path = binary_path
        value = self.env['ASE_ESPRESSO_COMMAND']
        self.env['ASE_ESPRESSO_COMMAND'] = value.format(binarypath=self.binary_path)
        self.pseudo_dir = pseudo_dir
        if pseudos is None:
            self.pseudos = rutgers_pseudos
        else:
            self.pseudos = pseudos
        self.job_root = job_root
        self.run_dir = run_dir

        os.environ.update(self.env)


def get_dune_qe_settings(run_dir_root: Path, material: str) -> EspressoSettings:
    """

    """
    dune3 = {'binary_path': '/users/sol/abuccheri/packages/qe-7.1/build/bin',
             'pseudo_dir': '/users/sol/abuccheri/rutgers_pseudos/pbesol',
             'pseudos': rutgers_pseudos[material],
             'run_dir': run_dir_root / material
             }
    env_settings = EspressoSettings(**dune3)
    return env_settings


# For the Rutgers pseudos
qe_n_electrons = {'silicon': 8,
                  'germanium': 28,
                  'diamond': 8,
                  'zinc_oxide': 52,
                  'mos2': 52,
                  'ws2': 52,
                  'bn_hex': 16,
                  'bn_cubic': 8,
                  'mgo': 16,
                  'nacl': 16,
                  'zro2': 24,
                  'pbs': 10,
                  'tio2_rutile': 48,
                  'tio2_ana': 48,
                  'cdse': 36,
                  'gan': 24,
                  'graphite': 16,
                  'gaas': 24,
                  'wo3_monoclinic': 128
                  }

# Number of conduction bands to add
n_c = 8
# Number of bands to compute per system
# Required, else we either pick arbitrary high constant or risk missing bandgaps.
number_of_bands = {m: int(0.5 * n_e) + n_c for m, n_e in qe_n_electrons.items()}
