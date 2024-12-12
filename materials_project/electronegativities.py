"""

"""
# Could also use Allen electronegativities:
# https://en.wikipedia.org/wiki/Electronegativity


# Extracted from the table: https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)
pauling = {
    'H': 2.20,
    'He': None,
    'Li': 0.98,
    'Be': 1.57,
    'B': 2.04,
    'C': 2.55,
    'N': 3.04,
    'O': 3.5,
    'F': 3.98,
    'Ne': None,
    'Na': 0.93,
    'Mg': 1.31,
    'Al': 1.61,
    'Si': 1.90,
    'P': 2.19,
    'S': 2.58,
    'Cl': 3.16,
    'Ar': None,
    'K': 0.82,
    'Ca': 1.00,
    'Sc': 1.36,
    'Ti': 1.54,
    'V': 1.63,
    'Cr': 1.66,
    'Mn': 1.55,
    'Fe': 1.83,
    'Co': 1.88,
    'Ni': 1.91,
    'Cu': 1.90,
    'Zn': 1.65,
    'Ga': 1.81,
    'Ge': 2.01,
    'As': 2.18,
    'Se': 2.55,
    'Br': 2.96,
    'Kr': 3.00,
    'Rb': 0.82,
    'Sr': 0.95,
    'Y': 1.22,
    'Zr': 1.33,
    'Nb': 1.6,
    'Mo': 2.16,
    'Tc': 1.9,
    'Ru': 2.2,
    'Rh': 2.28,
    'Pd': 2.20,
    'Ag': 1.93,
    'Cd': 1.69,
    'In': 1.78,
    'Sn': 1.96,
    'Sb': 2.05,
    'Te': 2.1,
    'I': 2.66,
    'Xe': 2.6,
    'Cs': 0.79,
    'Ba': 0.89,
    'La': 1.10,
    'Ce': 1.12,
    'Pr': 1.13,
    'Nd': 1.14,
    'Pm': None,
    'Sm': 1.17,
    'Eu': None,
    'Gd': 1.20,
    'Tb': None,
    'Dy': 1.22,
    'Ho': 1.23,
    'Er': 1.24,
    'Tm': 1.25,
    'Yb': None,
    'Lu': 1.27,
    'Hf': 1.3,
    'Ta': 1.5,
    'W': 2.36,
    'Re': 1.9,
    'Os': 2.2,
    'Ir': 2.20,
    'Pt': 2.28,
    'Au': 2.54,
    'Hg': 2.00,
    'Tl': 1.62,
    'Pb': 2.33,
    'Bi': 2.02,
    'Po': 2.0,
    'At': 2.2,
    'Rn': None,
    'Fr': None,
    'Ra': 0.9,
    'Ac': 1.1,
    'Th': 1.3,
    'Pa': 1.5,
    'U': 1.38,
    'Np': 1.36,
    'Pu': 1.28,
    'Am': 1.3,
    'Cm': 1.3,
    'Bk': 1.3,
    'Cf': 1.3,
    'Es': 1.3,
    'Fm': 1.3,
    'Md': 1.3,
    'No': 1.3
}
