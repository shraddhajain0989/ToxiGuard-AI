import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
tox = pd.read_csv('data/cleaned_tox21.csv')
zinc = pd.read_csv('data/zinc250k.csv')

print("Tox21:", tox.shape)
print("ZINC:", zinc.shape)

# Check columns
print(zinc.columns)

# Use logP if available
if 'logP' in zinc.columns:
    plt.hist(zinc['logP'], bins=50)
    plt.title("ZINC logP Distribution")
    plt.xlabel("logP")
    plt.ylabel("Frequency")
    plt.savefig('app/static/zinc_logp.png')
    print("ZINC logP plot saved ✅")

# Count molecules
plt.bar(['Tox21', 'ZINC'], [len(tox), len(zinc)])
plt.title("Dataset Size Comparison")
plt.savefig('app/static/dataset_comparison.png')

print("Analysis complete ✅")