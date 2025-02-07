import warnings
from typing import List

from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Parses the requirements.txt file and returns a list of requirements,
    excluding comments, empty lines, and the editable installation directive (`-e .`).

    Note: `-e .` is skipped because editable installations are intended for local development
    and may cause portability issues in CI/CD or production environments.
    """
    try:
        with open(file_path, encoding="utf-8") as file_obj:
            # Read and clean lines, skipping comments and empty lines
            requirements = [
                req.strip()
                for req in file_obj
                if req.strip()
                and not req.strip().startswith("#")
                and req.strip() != HYPHEN_E_DOT
            ]
        return requirements
    except FileNotFoundError:
        warnings.warn(
            f"Warning: {file_path} not found. Proceeding without additional dependencies.",
            stacklevel=2,
        )
        return []
    except UnicodeDecodeError as e:
        warnings.warn(
            f"Warning: Encoding issue while reading {file_path}: {str(e)}. Proceeding without additional dependencies.",
            stacklevel=2,
        )
        return []


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml_project_template",
    version="0.0.1",
    author="kjpou1",
    author_email="wasssssuuuuupppp@gmail.com",
    description="A modular and production-ready template for ML projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning template production-ready",
    url="https://github.com/kjpou1/ml_project_template",
    project_urls={
        "Bug Tracker": "https://github.com/kjpou1/ml_project_template/issues",
        "Documentation": "https://github.com/kjpou1/ml_project_template",
    },
    python_requires=">=3.7",
)
