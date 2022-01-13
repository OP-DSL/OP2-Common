# Standard library
from setuptools import setup

# TODO: Finish
setup(
    name="opcg",
    version="1.0.0",
    packages=["opcg"],
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["opcg=opcg.__main__:main"]},
)
