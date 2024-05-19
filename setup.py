from setuptools import setup, find_packages

from pypaq.lipytools.files import get_requirements


setup(
    name=               'pypaq',
    version=            'v1.7.7',
    url=                'https://github.com/piteren/pypaq.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    description=        'little Python tools',
    packages=           find_packages(),
    install_requires=   get_requirements(),
    license=            'MIT')