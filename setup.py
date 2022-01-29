from setuptools import setup, find_packages


def read_requirements():
    try:
        with open("requirements.txt", encoding="utf8") as fd:
            return fd.read()
    except TypeError:
        with open("requirements.txt") as fd:
            return fd.read()


setup(
    name="patchflow",
    version="0.1.0",
    description="A data generator for remote sensing segmentation models.",
    author="Roberto Nogueras",
    author_email="rnogueras@protonmail.com",
    url="https://github.com/rnogueras/patchflow",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=read_requirements().splitlines(),
)
