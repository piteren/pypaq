from setuptools import setup, find_packages

# reads requirements
def get_requirements():
    with open('requirements.txt') as file:
        lines = [l[:-1] if l[-1]=='\n' else l for l in file.readlines()]
        return lines


setup(
    name=               'pypaq',
    url=                'https://github.com/piteren/pypaq.git',
    author=             'Piotr Niewinski',
    author_email=       'pioniewinski@gmail.com',
    packages=           find_packages(),
    version=            'v0.1.1',
    install_requires=   get_requirements(),
    license=            'MIT',
    description=        'python tools')