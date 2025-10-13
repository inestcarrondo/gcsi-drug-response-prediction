from setuptools import setup, find_packages

setup(
    name="data_science",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "rdkit-pypi>=2022.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    author="Data Science Team",
    description="gCSI Drug Response Prediction Package",
)