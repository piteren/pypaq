import os
import unittest

from tests.envy import get_tmp_dir

from pypaq.lipytools.little_methods import prep_folder

EXCLUDE_DIRS = [
    '__pycache__',
    #'mpython',
    #'lipytools',
    #'pms',
    #'hpmser',
    #'neuralmess',
    #'neuralmess_duo'
]


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_dirs = [d for d in os.listdir(script_dir) if os.path.isdir(f'{script_dir}/{d}')]
    start_dirs = [d for d in start_dirs if d not in EXCLUDE_DIRS]
    print(f'\nTESTS starts for directories: {" | ".join(start_dirs)}')

    prep_folder(get_tmp_dir(), flush_non_empty=True)

    for start_dir in start_dirs:
        print(f'\n ************************ {start_dir} TESTS:')
        loader = unittest.TestLoader()
        suite = loader.discover(f'{script_dir}/{start_dir}')
        runner = unittest.TextTestRunner()
        runner.run(suite)