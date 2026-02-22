from setuptools import setup, find_packages

setup(
    name="credit-risk",
    version="1.0.0",
    description="Loan Default Risk Modeling — end-to-end credit risk pipeline",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.12.0",
        "shap>=0.44.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    ],
)
