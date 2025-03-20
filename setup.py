from setuptools import find_packages, setup

# OBS: "flash-attn==2.5.8" precisa ser instalado depois pq ele requer
# que alguns pacotes estejam instalados antes como wheel, ninja e packaging
# instala ele depois com pip install flash-attn==2.5.8 --no-build-isolation

_deps = [
    "numpy<=1.26.4",
    "charylu-tokenizer==0.0.6",
    "tqdm",
    "torch<=2.3.1",
    "wheel",
    "bitsandbytes<=0.43.1",
    "packaging<=24.1",
    "ninja<=1.11.1.1",
    "torchmetrics<=1.4.0.post0",
    "deepspeed<=0.14.4",
]

setup(
    name="charylu-models",
    packages=find_packages(include=["charylumodels"]),
    # include_package_data=True,
    package_data={"": ["transformer/*.py", "models/**/*.py"]},
    version="0.0.6",
    description="Biblioteca de modelos implemantados por Luis Chary",
    author="Luis Felipe Chary",
    install_requires=_deps,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    extras_require={"lightning": ["lightning<=2.3.0"]},
)
