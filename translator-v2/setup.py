from setuptools import setup

# TODO: Finish
setup(
    name="op2-translator",
    version="1.0.0",
    packages=["op2-translator"],
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["op2-translator=op2-translator.__main__:main"]},
)
