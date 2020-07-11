import sys
from os import path

from setuptools import find_packages, setup


PY36 = "py36"
PY37 = "py37"
PY38 = "py38"

if sys.version_info >= (3, 6, 8):
    py_tag = PY36
elif sys.version_info >= (3, 8, 0):
    py_tag = PY38
elif sys.version_info >= (3, 7, 0):
    py_tag = PY37
else:
    raise OSError(
        "Rosetta requires Python 3.6.8 and above, but yours is %s" % sys.version
    )


try:
    pkg_name = "rosetta_stone"
    pkg_slug = "rosetta"
    libinfo_py = path.join(pkg_slug, "__init__.py")
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

# remove blank lines and comments
base_dep = [
    x.strip()
    for x in base_dep
    if ((x.strip()[0] != "#") and (len(x.strip()) > 3) and "-e git://" not in x)
]

extras_dep = {"chinese": ["jieba"], "audio": ["librosa>=0.7.0", "torchaudio==0.4.0"]}


# def combine_dep(new_key, base_keys):
#     extras_dep[new_key] = list(set(k for v in base_keys for k in extras_dep[v]))


# combine_dep("nlp", ["transformers"])
# combine_dep("cn_nlp", ["chinese", "nlp"])
# combine_dep("all", [k for k in extras_dep if k != "elmo"])

setup(
    name=pkg_name,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test", "docs", "examples"]
    ),
    version=__version__,
    include_package_data=True,
    author="numb3r3",
    author_email="wangfelix87@gmail.com",
    description="make research work more friendly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/numb3r3/rosetta_stone",
    install_requires=base_dep,
    extras_require=extras_dep,
    setup_requires=[
        "setuptools>=18.0",
        "pytest-runner",
        "black==19.3b0",
        "isort==4.3.21",
    ],
    tests_require=["pytest"],
    python_requires=">=3.6",
    entry_points={"console_scripts": ["rosetta=rosetta.main.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
