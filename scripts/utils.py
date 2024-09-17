import subprocess
import pickle
from datetime import datetime


def get_git_hash() -> str:
    """Get git hash as string"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return function_wrapper


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        filename (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        Dictionary: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum
