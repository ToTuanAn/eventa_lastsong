import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely."""
    content = ""
    with io.open(
            os.path.join(os.path.dirname(__file__), *paths),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    """Parse requirements from a file."""
    if not os.path.exists(path):
        return []
    return [
        line.strip()
        for line in read(path).split("\n")
        if line and not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="eventa",
    description="Awesome eventa package",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src*"]),
    install_requires=read_requirements("requirements.txt"),
)