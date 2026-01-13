# # load_and_preprocess.py
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import json

# # Step 1: Load the dataset
# df = pd.read_csv('synthetic_cystic_fibrosis_dataset.csv')

# # Step 2: Assign column names (based on your schema)
# columns = [
#     'age_months', 'family_history_cf', 'parental_carrier_status', 'ethnicity',
#     'newborn_screening_result', 'salty_skin', 'cough_type', 'cough_character',
#     'respiratory_infections_frequency', 'wheezing_present', 'nasal_polyps',
#     'clubbing_fingers', 'weight_percentile', 'height_percentile', 'weight_for_height',
#     'growth_faltering', 'appetite', 'stool_character', 'stool_frequency',
#     'abdominal_distention', 'diarrhea_chronic', 'fat_malabsorption_signs',
#     'meconium_ileus', 'prolonged_jaundice', 'failure_to_thrive',
#     'sweat_test_simulated', 'respiratory_score', 'nutritional_risk_score',
#     'cf_clinical_suspicion_index', 'cf_diagnosis', 'diagnostic_confidence',
#     'age_at_diagnosis'
# ]

# if len(df.columns) == len(columns):
#     df.columns = columns
# else:
#     raise ValueError(f"Column count mismatch: expected {len(columns)}, got {len(df.columns)}")

# print("✅ Columns assigned successfully.")
# print(f"Dataset shape: {df.shape}")
# print(f"CF prevalence: {df['cf_diagnosis'].mean():.4f} ({df['cf_diagnosis'].sum()} cases)")

# # Step 3: Define target: who needs sweat test referral?
# df['needs_sweat_test'] = (df['cf_clinical_suspicion_index'] >= 40).astype(int)

# # Step 4: Select feature columns (exclude diagnosis-related leakage)
# feature_cols = [
#     'age_months', 'family_history_cf', 'ethnicity', 'salty_skin',
#     'cough_type', 'respiratory_infections_frequency', 'wheezing_present',
#     'weight_percentile', 'growth_faltering', 'stool_character',
#     'meconium_ileus', 'sweat_test_simulated', 'respiratory_score',
#     'nutritional_risk_score'
# ]

# # Step 5: Encode categorical variables
# df['ethnicity'] = pd.Categorical(df['ethnicity']).codes
# df['stool_character'] = pd.Categorical(df['stool_character']).codes

# # Step 6: Handle missing values (none expected, but safe)
# X = df[feature_cols].fillna(0)
# y = df['needs_sweat_test']

# # Step 7: Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Save for FL clients
# np.save('X.npy', X_scaled)
# np.save('y.npy', y.values)

# # Step 8: Create non-IID client partitions by ethnicity
# client_data = {}
# ethnicities = df['ethnicity'].unique()
# for i, eth in enumerate(ethnicities):
#     indices = df[df['ethnicity'] == eth].index.tolist()
#     if len(indices) > 100:
#         client_data[f'client_{i}'] = indices

# # Save partition mapping
# with open('client_partitions.json', 'w') as f:
#     json.dump({k: v for k, v in client_data.items()}, f)

# print(f"\n✅ Created {len(client_data)} non-IID clients based on ethnicity.")
# for k, v in client_data.items():
#     print(f"  {k}: {len(v)} samples")


# # Add to end of load_and_preprocess.py
# print("Suspicion index stats:")
# print(df['cf_clinical_suspicion_index'].describe())
# print("\nHigh suspicion (>=40):", (df['cf_clinical_suspicion_index'] >= 40).sum())

# # Identify high-suspicion cases
# high_risk = df[df['cf_clinical_suspicion_index'] >= 40]
# low_risk = df[df['cf_clinical_suspicion_index'] < 40]

# # If too few high-risk, duplicate them (simple oversampling)
# if len(high_risk) < 100:
#     # Repeat high-risk cases 10x to ensure enough signal
#     high_risk_balanced = pd.concat([high_risk] * max(10, 100 // len(high_risk)), ignore_index=True)
# else:
#     high_risk_balanced = high_risk

# # Combine with low-risk (optionally downsample low-risk if too large)
# balanced_df = pd.concat([high_risk_balanced, low_risk.sample(n=min(5000, len(low_risk)), random_state=42)], ignore_index=True)

# # Shuffle
# balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Use balanced_df instead of df for rest of pipeline
# df = balanced_df

# load_and_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Load dataset
df = pd.read_csv('synthetic_cystic_fibrosis_dataset.csv')

# Assign column names
columns = [
    'age_months', 'family_history_cf', 'parental_carrier_status', 'ethnicity',
    'newborn_screening_result', 'salty_skin', 'cough_type', 'cough_character',
    'respiratory_infections_frequency', 'wheezing_present', 'nasal_polyps',
    'clubbing_fingers', 'weight_percentile', 'height_percentile', 'weight_for_height',
    'growth_faltering', 'appetite', 'stool_character', 'stool_frequency',
    'abdominal_distention', 'diarrhea_chronic', 'fat_malabsorption_signs',
    'meconium_ileus', 'prolonged_jaundice', 'failure_to_thrive',
    'sweat_test_simulated', 'respiratory_score', 'nutritional_risk_score',
    'cf_clinical_suspicion_index', 'cf_diagnosis', 'diagnostic_confidence',
    'age_at_diagnosis'
]

if len(df.columns) == len(columns):
    df.columns = columns
else:
    raise ValueError("Column count mismatch")

print("✅ Columns assigned.")
print(f"Original shape: {df.shape}")
print(f"CF cases: {df['cf_diagnosis'].sum()}")
print(f"High suspicion (>=40): {(df['cf_clinical_suspicion_index'] >= 40).sum()}")

# Define target: needs sweat test referral
df['needs_sweat_test'] = (df['cf_clinical_suspicion_index'] >= 40).astype(int)

# Separate high and low suspicion
high_risk = df[df['needs_sweat_test'] == 1]
low_risk = df[df['needs_sweat_test'] == 0]

print(f"\nBefore balancing: High={len(high_risk)}, Low={len(low_risk)}")

# If too few high-risk, oversample them
if len(high_risk) < 100:
    # Repeat high-risk samples to get at least 200
    high_risk_balanced = pd.concat([high_risk] * max(1, 200 // len(high_risk)), ignore_index=True)
else:
    high_risk_balanced = high_risk

# Downsample low-risk to avoid extreme imbalance (e.g., keep 2000)
low_risk_balanced = low_risk.sample(n=min(2000, len(low_risk)), random_state=42)

# Combine
balanced_df = pd.concat([high_risk_balanced, low_risk_balanced], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"After balancing: High={len(balanced_df[balanced_df['needs_sweat_test']==1])}, Low={len(balanced_df[balanced_df['needs_sweat_test']==0])}")

# Select features
feature_cols = [
    'age_months', 'family_history_cf', 'ethnicity', 'salty_skin',
    'cough_type', 'respiratory_infections_frequency', 'wheezing_present',
    'weight_percentile', 'growth_faltering', 'stool_character',
    'meconium_ileus', 'sweat_test_simulated', 'respiratory_score',
    'nutritional_risk_score'
]

# Encode categoricals
balanced_df['ethnicity'] = pd.Categorical(balanced_df['ethnicity']).codes
balanced_df['stool_character'] = pd.Categorical(balanced_df['stool_character']).codes

# Handle missing values
X = balanced_df[feature_cols].fillna(0)
y = balanced_df['needs_sweat_test']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save
np.save('X.npy', X_scaled)
np.save('y.npy', y.values)

# Create non-IID client partitions (by ethnicity)
client_data = {}
eth_groups = balanced_df['ethnicity'].unique()
for i, eth in enumerate(eth_groups):
    indices = balanced_df[balanced_df['ethnicity'] == eth].index.tolist()
    if len(indices) > 50:  # ensure min size
        client_data[f'client_{i}'] = indices

with open('client_partitions.json', 'w') as f:
    json.dump({k: v for k, v in client_data.items()}, f)

print(f"\n✅ Created {len(client_data)} clients.")
for k, v in client_data.items():
    print(f"  {k}: {len(v)} samples")