from setuptools import setup

setup(name='tb_benchmarking',
      version='0.0.1',
      description='Utilities for generating, running and processing TB lite inputs/outputs',
      author='Alex Buccheri',
      author_email='alexanderbuccheri@googlemail.com',
      packages=['tb_lite'],
      install_requires=[
          'numpy>=1.14.5',
          'matplotlib>=2.2.0',
          'ase>=3.22.0',
          'pymatgen',
          'seekpath>=2.0.1',
          'pytest',
          'toml']
      )
