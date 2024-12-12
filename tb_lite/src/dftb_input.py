""" Generate DFTB+/TB lite input files
"""
from typing import Union
import numpy as np

from tb_lite.src.band_utils import preprocess_ase_path


class DftbInput:
    """ Class to collate all DFTB+ input classes
    """

    def __init__(self,
                 driver=None,
                 hamiltonian=None,
                 options=None):
        self.driver = driver
        self.hamiltonian = hamiltonian
        self.options = options

    def generate_dftb_hsd(self) -> str:
        """ Generate dftb_in.hsd input string

        GEN format done separately, with ASE
        """
        driver_string = self.driver.to_string() if isinstance(self.driver, Driver) else ''
        ham_string = self.hamiltonian.to_string() if isinstance(self.hamiltonian, (Hamiltonian, BandStructureHamiltonian)) else ''
        options_string = self.options.to_string() if isinstance(self.options, Options) else ''

        dftb_in_template = f"""
    Geometry = GenFormat {{
        <<< "geometry.gen"
    }}

    {driver_string}

    {ham_string}

    {options_string}

    Parallel {{
      UseOmpThreads = Yes
    }}
        """
        return dftb_in_template


class Driver:
    def __init__(self,
                 type='ConjugateGradient',
                 lattice_option='No'):
        self.type = type
        self.lattice_option = lattice_option

    def to_string(self):
        string = f"""Driver = {self.type} {{
  LatticeOpt = {self.lattice_option}
}}"""
        return string


class KGrid:
    def __init__(self, k_grid: list, k_weights=None):
        """ k-grid sampling class
        :param k_grid: 3 integers
        """
        self.k_grid = k_grid
        self.k_weights = self.set_k_weights(k_weights)

    def set_k_weights(self, k_weights) -> list:
        if k_weights is not None:
            return k_weights

        # 0.0 if odd, 0.5 if even
        weights = [0.0, 0.0, 0.0]
        is_even = lambda x: x % 2 == 0
        for i, k in enumerate(self.k_grid): 
            if is_even(k): weights[i] = 0.5
        
        return weights

    def to_string(self) -> str:
        k1, k2, k3 = self.k_grid
        w1, w2, w3 = self.k_weights
        string = f"""  KPointsAndWeights = SuperCellFolding {{
    {k1} 0 0
    0 {k2} 0
    0 0 {k3}
    {w1} {w2} {w3}
  }}
        """
        return string


class ExplicitKPoints:
    def __init__(self, k_points: np.ndarray, k_weights=None):
        """

        :param k_points:  Expect .shape = (n_k_points, 3)
        :param k_weights:
        """
        self.k_points = k_points
        self.n_k_points = k_points.shape[0]
        self.k_weights = self.set_k_weights(k_weights)

    def set_k_weights(self, k_weights) -> list:
        if k_weights is None:
            return [1.0] * self.n_k_points
        else:
            return k_weights

    def to_string(self) -> str:
        string = "  KPointsAndWeights = {\n"
        weight = "1.0"

        for ik in range(0, self.n_k_points):
            k_str = np.array2string(self.k_points[ik, :], precision=8, separator=' ', suppress_small=False)[1:-1]
            string += "    " + k_str + " " + weight + "\n"

        string += "    }\n"

        return string


class KLines:
    def __init__(self, bandpath, points_per_path):
        self.bandpath = bandpath
        self.points_per_path = points_per_path

    def to_string(self):
        path = preprocess_ase_path(self.bandpath.path)
    
        string = "  KPointsAndWeights [relative] = KLines { \n"

        symbol = path[0]
        k_point = self.bandpath.special_points[symbol]
        k_point_str = " ".join(str(k) for k in k_point)
        string += f"    1  {k_point_str}  # {symbol} \n"

        points_per_path = self.points_per_path
        for i in range(1, len(path)):
            symbol = path[i]
            # Set a new starting point, following ',' in the path
            if symbol == ',':
                points_per_path = 1
                continue
            k_point = self.bandpath.special_points[symbol]
            k_point_str = " ".join(str(k) for k in k_point)
            string += f"    {points_per_path}  {k_point_str}  # {symbol} \n"
            points_per_path = self.points_per_path

        string += "    }\n"
        return string


class Hamiltonian:
    def __init__(self,
                 method='GFN1-xTB',
                 ParameterFile=None,
                 temperature=0.0,
                 scc_tolerance=1.e-6,
                 k_grid=None,
                 k_weights=None,
                 max_scf=50):
        if k_grid is None:
            k_grid = [4, 4, 4]
        self.method = method
        self.ParameterFile = ParameterFile
        self.temperature = temperature
        self.scc_tolerance = scc_tolerance
        # Should initialise outside class and pass in object if being proper
        self.k_grid = KGrid(k_grid, k_weights)
        self.max_scf = max_scf

    def set_k_grid(self, k_grid, k_weights=None):
        if type(k_grid) is np.ndarray:
            k_grid = k_grid.tolist()
        self.k_grid = KGrid(k_grid, k_weights)

    def _xtb_method(self):
        """ParameterFile takes precedence over method"""
        if self.ParameterFile:
            return f'ParameterFile = "{self.ParameterFile}"'
        else:
            return f'Method = "{self.method}"'

    def to_string(self):
        string = f"""Hamiltonian = xTB {{
  {self._xtb_method()}
  SCC = Yes
  SCCTolerance = {self.scc_tolerance}
  Filling = Fermi {{
    Temperature [Kelvin] = {self.temperature}
  }}
  {self.k_grid.to_string()}
}}
        """
        return string


class BandStructureHamiltonian:
    """
    I was originally using self.k_points = ExplicitKPoints(k_points, k_weights)
    to define the k-path, as this allows one to use ASE's points explicitly,
    but this requires ALOT of post-processing on the ASE bathpath to make it 
    useable. One is probably better-off using Klines
    """
    def __init__(self,
                 k_points: Union[ExplicitKPoints, KLines],
                 method='GFN1-xTB',
                 ParameterFile=None,
                 ):
        self.k_points = k_points
        self.method = method
        self.ParameterFile = ParameterFile

    def _xtb_method(self):
        """ParameterFile takes precedence over method"""
        if self.ParameterFile:
            return f'ParameterFile = "{self.ParameterFile}"'
        else:
            return f'Method = "{self.method}"'

    def to_string(self):
        string = f"""Hamiltonian = xTB {{
  {self._xtb_method()}
  SCC = Yes
  ReadInitialCharges = Yes
  MaxSCCIterations = 1
  {self.k_points.to_string()}
}}
        """
        return string


class Options:
    def __init__(self, timing_level=1):
        self.timing_level = timing_level

    def to_string(self):
        string = f"""Options = {{
   TimingVerbosity = {self.timing_level}
}}"""
        return string


# Depreciated in favour of injection
# def generate_band_structure_input(lattice_vectors, method: str, npoints: int) -> str:
#     """ Generate DFTB+ input file string for a band structure calculation,
#     using a band path as standardised by ASE.
#
#     :param lattice_vectors: Crystal lattice vectors
#     :param method: Calculation method
#     :return: Input file string
#     """
#     assert method in ['GFN1-xTB', 'GFN2-xTB'], "Method is not valid"
#     band_path = get_standardised_band_path_object(lattice_vectors, npoints=npoints)
#     h_bands = BandStructureHamiltonian(band_path.kpts, method=method)
#     return DftbInput(hamiltonian=h_bands).generate_dftb_hsd()
