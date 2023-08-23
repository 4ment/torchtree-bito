import os

from setuptools import setup

with open(os.path.join("torchtree_bito", "_version.py")) as f:
    __version__ = f.readlines()[-1].split()[-1].strip("\"'")

if __name__ == '__main__':
    kwargs = {
        'version': __version__,
    }
    setup(**kwargs)
