from setuptools import find_packages, setup

setup(
    name="charylu-models",
    packages=find_packages(include=["charylumodels"]),
    include_package_data=True,
    version="0.0.1",
    description="Biblioteca de modelos implemantados por Luis Chary",
    author="Luis Felipe Chary",
    install_requires=["torch==2.3.1", "numpy==2.0.0"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
