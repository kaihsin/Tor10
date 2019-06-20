# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tor10",
    version="0.3.7",
    author="Kai-Hsin Wu",
    author_email="author@example.com",
    description="Tensor network framework based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaihsin/Tor10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','torch'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
)
