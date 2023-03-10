from setuptools import setup, find_packages

def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires

requires = open_requirements('requirements.txt')
setup(
    name='pykefield',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description=('A python-based KEMField electric field '
                 'simulation analysis framework'),
    long_description=open('README.md').read(),
    install_requires=requires,
    url='https://github.com/ricmperes/pykefield',
    author='Ricardo Peres',
    author_email='rperes@physik.uzh.ch'
)