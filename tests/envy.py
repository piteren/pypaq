import os

from pypaq.lipytools.little_methods import prep_folder


def get_tmp_dir() -> str:
    tests_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(tests_dir, '_tmp')

def flush_tmp_dir() -> str:
    tmp_dir = get_tmp_dir()
    prep_folder(tmp_dir, flush_non_empty=True)
    return tmp_dir


if __name__ == "__main__":
    print(flush_tmp_dir())