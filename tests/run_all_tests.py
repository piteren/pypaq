import os
import unittest

from tests.envy import flush_tmp_dir

EXCLUDE_DIRS = [
    '__pycache__',
    #'mpython',
]


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    start_dirs = [d for d in os.listdir(script_dir) if os.path.isdir(f'{script_dir}/{d}')]
    start_dirs = [d for d in start_dirs if d not in EXCLUDE_DIRS]
    print(f'\nTESTS starts for directories: {" | ".join(start_dirs)}')

    flush_tmp_dir()

    for start_dir in start_dirs:
        print(f'\n ************************ {start_dir} TESTS:')
        loader = unittest.TestLoader()
        suite = loader.discover(f'{script_dir}/{start_dir}')
        runner = unittest.TextTestRunner()
        runner.run(suite)