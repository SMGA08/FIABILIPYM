from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="fiabilipym",
    version="2.8.1",
    author="chabotsi, crubier, cdrom1",
    author_email="contact@fiabilipy.org",
    description="fiabilipy with Python 3 compatibility and graph visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://fiabilipy.org",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "matplotlib>=3.9.4",
        "networkx>=3.2.1",
        "numpy>=1.26.4",
        "pygraphviz>=1.11",
        "scipy>=1.13.1",
        "sympy>=1.14.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
