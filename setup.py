from setuptools import setup, find_packages

setup(
    name="Adv_ML",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "torch", "numpy",
                      "matplotlib", "scikit-learn", "scipy",
                      "tqdm"],
)