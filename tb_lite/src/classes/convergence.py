""" Convergence class.
"""
from typing import Callable, Optional, List
import numpy as np


class Convergence:
    """ Store inputs, outputs and their differences.
    """
    # Indexing to start at
    first_iteration = 0

    @staticmethod
    def default_diff(iteration, output):
        return abs(output[iteration] - output[iteration - 1])

    def __init__(self, target_delta: float, max_iter=None, diff_func: Optional[Callable] = None):
        self.target_delta = target_delta
        self.input = []
        self.output = []
        self.max_iter = max_iter
        self.iteration = self.first_iteration - 1
        self.converged = False
        # Need to think about how to make the API flexible if injecting a diff function
        self.diff_funct = self.default_diff if diff_func is None else diff_func

    def diff(self, iteration):
        # Cannot evaluate diff on first iteration
        if self.iteration <= self.first_iteration:
            return np.nan
        return self.diff_funct(iteration, self.output)

    def update(self, input, output):
        self.input.append(input)
        self.output.append(output)
        self.iteration += 1
        self.converged = self.diff(self.iteration) <= self.target_delta

    def has_converged(self) -> bool:
        return self.converged

    def iterations_exceeded(self) -> bool:
        if self.max_iter is None:
            return False
        # Account for zero-indexing
        return self.iteration > self.max_iter - 1

    def _print(self, iteration: int):
        out_of = ''
        if self.max_iter is not None:
            out_of = f'/ {self.max_iter}'

        print(f'Iteration, Input, output, diff, target diff: '
              f'{iteration} {out_of}, {self.input[iteration]}, {self.output[iteration]}, '
              f'{self.diff(iteration)}, {self.target_delta}')

    def print_iteration(self):
        self._print(self.iteration)

    def print_summary(self):
        converge_str = 'converged'
        if not self.converged:
            converge_str = 'has not ' + converge_str

        print(f'Convergence Summary. Calculation {converge_str}:')
        for i in range(self.first_iteration + 1, len(self.input)):
            self._print(i)

    def serialise(self, input_label, output_label) -> List[dict]:
        results = []
        for i in range(self.first_iteration , len(self.input)):
            results.append({input_label: self.input[i],
                            output_label: self.output[i]}
                            )
        return results
