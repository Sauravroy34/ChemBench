from setuptools import setup, find_packages

setup(
    name="ChemLLMBench",
    version="0.1.0",
    description="Chemistry LLM Benchmark Suite",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ChemLLMBench": [
            "Name_Prediction/*.csv",
            "Reaction_prediction/*.csv",
            "Retrosynthesis/*.csv",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "rdkit",
        "scikit-learn",
        "litellm",
        "python-dotenv",
        "requests",
    ],
)
