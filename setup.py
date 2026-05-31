from setuptools import setup, find_packages

setup(
    name="eamsnet-cd",
    version="1.0.0",
    description="EAMSNet: change detection on LEVIR-CD with ATDAM/MSDA/EABRM modules",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12", "torchvision>=0.13", "numpy>=1.21",
        "pillow>=8.0", "matplotlib>=3.4", "pyyaml>=5.4", "scipy>=1.7",
    ],
)
