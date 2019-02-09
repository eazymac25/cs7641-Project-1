from setuptools import setup, find_packages

setup(
    name='kmacneney3-project1-solution',
    version='1.0.0',
    description='Classification algorithm evaluations',
    url='https://github.com/eazymac25/cs7641-Project-1',
    author='Kyle MacNeney',
    author_email='kyle.macneney@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=['exploratory_notebooks']),
    install_requires=[
        'requests',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'python-graphviz',
        'graphviz',
        'seaborn',
    ],
)
