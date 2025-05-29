import subprocess
import sys
import importlib

def install_if_missing(packages):
    for package in packages:
        try:
            # Try to import the package to check if installed
            importlib.import_module(package)
            print(f"'{package}' is already installed.")
        except ImportError:
            print(f"'{package}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("Done checking all packages.")

# Note: For some packages, the import name may differ from the package name in pip
# If needed, create a mapping like: {'tensorflow': 'tensorflow', 'sklearn': 'scikit-learn', ...}

# Define your packages here (pip package names)
packages_to_install = [
    "scikit-learn",
    "tensorflow",
    "streamlit",
]

# For import checking, use the correct import names here
import_names = [
    "sklearn",      # for scikit-learn
    "tensorflow",
    "streamlit",
]

# Run installation check
for pkg, imp_name in zip(packages_to_install, import_names):
    try:
        importlib.import_module(imp_name)
        print(f"'{imp_name}' is already installed.")
    except ImportError:
        print(f"'{imp_name}' not found. Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

print("All done!")
