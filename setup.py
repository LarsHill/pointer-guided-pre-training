import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(file_name: str, pkg="llm"):
    spec = spec_from_file_location(os.path.join(pkg, file_name), os.path.join(_PATH_ROOT, pkg, file_name))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_deps = ["pytest>=7.0.0", "pytest-cov", "black>=22.6", "pre-commit>=2.17.0"]

setup(
    name="llm",
    version=about.__version__,
    description=about.__docs__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about.__author__,
    license=about.__license__,
    copyright=about.__copyright__,
    keywords=["language modeling", "machine-learning", "deep-learning"],
    packages=find_packages(),
    python_requires=">=3.10.0",
    package_data={"": ["*.txt"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: IPython",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Typing :: Typed",
    ],
    install_requires=[
        "beautifulsoup4",
        "datasets>=2.10",
        "einops",
        "fluidml>=0.3.0",
        "html5lib",
        "lxml",
        "markdown",
        "matplotlib",
        "pymongo",
        "seaborn",
        "numpy",
        "pandas",
        "lightning",
        "pyyaml",
        "requests",
        "scikit-learn",
        "somajo",  # GPL 3
        "tenacity",
        "tokenizers",
        "torch>=2.0",
        "torchmetrics",
        "tqdm",
        "transformers>=4.27",
        "wandb",
        "wikimapper",
    ],
    extras_require={
        "tests": test_deps,
    },
    tests_require=test_deps,
)
