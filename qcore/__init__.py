from qcoreutils.src.utils import Set
import collections

default_settings = collections.OrderedDict([
    ('h0_cutoff', Set(50, 'bohr')),
    ('overlap_cutoff', Set(50, 'bohr')),
    ('repulsive_cutoff', Set(50, 'bohr')),
    # Ewald setting for hard-cutoff of potential at 30 bohr
    ('ewald_real_cutoff', Set(40, 'bohr')),
    ('overlap_screening_threshold', Set(-1)),
    # Converged w.r.t. real-space value
    ('ewald_reciprocal_cutoff', Set(10)),
    ('ewald_alpha', Set(0.3)),
    ('ewald_expansion_order', Set(13)),
    ('overlap_screening_threshold', Set(-1)),
    ('symmetry_reduction', Set(False)),
    ('temperature', Set(0, 'kelvin')),
    ('potential_type', Set('full')),
    ('solver', Set('SCC'))
])