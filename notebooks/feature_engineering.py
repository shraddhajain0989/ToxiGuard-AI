import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

df = pd.read_csv('data/cleaned_tox21.csv')

print("Loaded:", df.shape)

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    # Enhanced Morgan Fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    fp_array = np.array(fp)
    
    # MACCS Keys
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_array = np.array(maccs)

    return np.concatenate([features, fp_array, maccs_array])


X = []
y = []

for i, smiles in enumerate(df['smiles']):
    if i % 500 == 0:
        print(f"Processing {i}/{len(df)}")

    feat = featurize(smiles)

    if feat is not None:
        X.append(feat)
        y.append(df['toxic'][i])

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

np.save('data/X.npy', X)
np.save('data/y.npy', y)

print("Features saved ✅")