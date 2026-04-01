import pandas as pd

# Load dataset
df = pd.read_csv('data/tox21.csv')

print("Original Shape:", df.shape)

# Step 1: Remove missing SMILES
df = df.dropna(subset=['smiles'])

# Step 2: Identify toxicity columns (exclude non-label columns)
non_tox_cols = ['smiles']
tox_cols = [col for col in df.columns if col not in non_tox_cols]

# Step 3: Convert all toxicity columns to numeric
df[tox_cols] = df[tox_cols].apply(pd.to_numeric, errors='coerce')

# Step 4: Convert multi-label → binary (ignore NaNs)
df['toxic'] = df[tox_cols].max(axis=1)

# Step 5: Drop untested molecules (where ALL labels were NaN)
df = df.dropna(subset=['toxic'])

# Cast explicitly to integer
df['toxic'] = df['toxic'].astype(int)

# Step 6: Keep only needed columns
df_clean = df[['smiles', 'toxic']]

print("Cleaned Shape:", df_clean.shape)
print(df_clean.head())

# Step 7: Save cleaned dataset
df_clean.to_csv('data/cleaned_tox21.csv', index=False)

print("Cleaned dataset saved ✅")