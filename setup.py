from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
# long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'conllu',
    'datasets>=2.9',
    'huggingface-hub>=0.12.0',
    'seqeval',
]
try:
    import transformers
except ImportError:
    install_requires.append('adapter-transformers==2.1.2')
setup(
    name='tlr-adapters',
    version='0.0.1',
    description='Tool for training target language-ready task adapters in PyTorch',
    url='https://github.com/parovicm/tlr-adapters',
    author='Marinela Parovic, Alan Ansell',
    author_email='marinelaparovic@gmail.com, aja63@cam.ac.uk',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires,
    python_requires='>=3.9',
)
