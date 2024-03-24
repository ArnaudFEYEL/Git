import importlib

required_packages = [
    "streamlit",
    "os",
    "time",
    "re",
    "numpy",
    "pandas",
    "subprocess",
    "matplotlib",
    "sys",
    "torch",
    "diffrax",
    "equinox",
    "jax",
    "optax",
    "math",
    "IPython.display",
    "tqdm",
    "seaborn"
]

missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"The following packages are missing: {', '.join(missing_packages)}")
else:
    print("All required packages are installed.")
