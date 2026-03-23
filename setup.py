from setuptools import setup

setup(
    name='mapping',
    version='0.0.1',
    author='Kristin Wu',
    description='Python mapping and scanning implementation and optimization.',
    long_description=open('README.md', encoding='utf-8').read(),
    packages=['scanning'],
    #package_data={'scanning': ['data']},
    #scripts=['bin/script1','bin/script2'],
    install_requires=[
        'astropy>=4.3.1',
        'fast-histogram>=0.10',
        'fyst-trajectories>=0.2.0',
        'matplotlib>=3.4.3',
        'numpy>=1.21.2',
        'pandas>=1.3.2',
        'scipy>=1.7.1',
        'notebook>=6.4.4'
    ],
)