from os import path

from setuptools import find_packages, setup


try:
    pkg_name = "rosetta"
    libinfo_py = path.join(pkg_name, "__init__.py")
    libinfo_content = open(libinfo_py, "r", encoding="utf8").readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith("__version__")][
        0
    ]
    exec(version_line)  # produce __version__
except FileNotFoundError:
    __version__ = "0.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    base_dep = f.read().splitlines()

setup(
    name=pkg_name,
    packages=find_packages(exclude=["test", "docs", "examples"]),
    version=__version__,
    include_package_data=True,
    author="numb3r3",
    author_email="wangfelix87@gmail.com",
    description="support tool for research experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.huya.com/wangfeng2/rosetta_stone",
    install_requires=base_dep,
    setup_requires=[
        "setuptools>=18.0",
        "pytest-runner",
        "black==19.3b0",
        "isort==4.3.21",
    ],
    tests_require=["pytest"],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
