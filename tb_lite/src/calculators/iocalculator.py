""" Abstract base class for an IO calculator
"""
from abc import ABC, abstractmethod


class IOCalculator(ABC):

    def __init__(self, directory):
        # Job directory
        self.directory = directory

    @abstractmethod
    def write_input(self) -> None:
        ...

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def parse_result(self) -> dict:
        ...
