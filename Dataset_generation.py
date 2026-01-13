import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
n_samples = 10000

# Step 1: Define ethnicity distribution and CF prevalence by group
ethnicities = ['Caucasian', 'Ashkenazi Jewish', 'Hispanic', 'African', 'Asian']
ethnicity_weights = [0.60, 0.05, 0.15, 0.10, 0.10]  # Approximate US mix

# Prevalence based on literature: ~1/2500 in Caucasians [[1]], lower in others
prevalence_map = {
    'Caucasian': 1 / 2500,
    'Ashkenazi Jewish': 1 / 2500,
    'Hispanic': 1 / 8000,
    'African': 1 / 15000,
    'Asian': 1 / 30000
}

# Step 2: Assign ethnicity
ethnicity = np.random.choice(ethnicities, size=n_samples, p=ethnicity_weights)

# Step 3: Assign CF diagnosis based on ethnicity-specific prevalence
cf_diagnosis = np.array([
    np.random.binomial(1, prevalence_map[eth]) for eth in ethnicity
])

# Step 4: Static Features
# Family history: ~30% of CF cases have known family history; background rate ~2%
family_history_cf = np.where(
    cf_diagnosis == 1,
    np.random.binomial(1, 0.30, n_samples),
    np.random.binomial(1, 0.02, n_samples)
)

# Parental carrier status (simplified)
def assign_carrier_status(fh, cf):
    if cf == 1:
        # If diagnosed, likely at least one carrier parent
        return np.random.choice(['both', 'one'], p=[0.25, 0.75])
    else:
        if fh == 1:
            return np.random.choice(['one', 'unknown'], p=[0.6, 0.4])
        else:
            return np.random.choice(['none', 'unknown'], p=[0.9, 0.1])

parental_carrier_status = np.array([
    assign_carrier_status(fh, cf) for fh, cf in zip(family_history_cf, cf_diagnosis)
])

# Newborn screening (assume 95% coverage in simulated setting)
newborn_screening_result = np.random.binomial(1, 0.95, n_samples)

# Step 5: Age in months (0–24 months, skewed toward younger for early detection)
age_months = np.random.gamma(shape=2.0, scale=4.0, size=n_samples).clip(0.5, 24).round().astype(int)

# Step 6: Generate sweat chloride levels
sweat_test_simulated = np.zeros(n_samples)
for i in range(n_samples):
    if cf_diagnosis[i] == 1:
        # CF: mean ~99, std ~30, truncated
        val = np.random.normal(99, 30)
        sweat_test_simulated[i] = np.clip(val, 30, 150)
    else:
        # Non-CF: mostly <40, some intermediate
        if np.random.rand() < 0.9:
            val = np.random.normal(25, 8)
        else:
            val = np.random.normal(45, 5)  # intermediate zone
        sweat_test_simulated[i] = np.clip(val, 10, 59)

# Step 7: Physical Symptoms
# Salty skin: highly specific [[24]]
salty_skin = np.where(
    cf_diagnosis == 1,
    np.random.binomial(1, 0.85, n_samples),
    np.random.binomial(1, 0.05, n_samples)
)

# Cough type (ordinal)
cough_type = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    if cf_diagnosis[i] == 1:
        probs = [0.10, 0.15, 0.40, 0.35]  # more severe in CF
    else:
        # Mimic asthma or viral infections
        probs = [0.70, 0.20, 0.08, 0.02]
    cough_type[i] = np.random.choice([0, 1, 2, 3], p=probs)

# Cough character
cough_character = np.empty(n_samples, dtype='U10')
for i in range(n_samples):
    if cough_type[i] == 0:
        cough_character[i] = 'none'
    elif cf_diagnosis[i] == 1:
        cough_character[i] = np.random.choice(['wet', 'productive'])
    else:
        cough_character[i] = np.random.choice(['dry', 'barking'])

# Respiratory infections (last 12 months)
respiratory_infections_frequency = np.where(
    cf_diagnosis == 1,
    np.random.poisson(2.5, n_samples).clip(0, 8),
    np.random.poisson(0.8, n_samples).clip(0, 5)
)

# Wheezing
wheezing_present = np.where(
    cf_diagnosis == 1,
    np.random.binomial(1, 0.60, n_samples),
    np.random.binomial(1, 0.15, n_samples)  # asthma mimic
)

# Nasal polyps (rare in infants, so age-dependent)
nasal_polyps = np.where(
    (cf_diagnosis == 1) & (age_months >= 12),
    np.random.binomial(1, 0.25, n_samples),
    np.random.binomial(1, 0.02, n_samples)
)

# Clubbing (very rare under 12 months)
clubbing_fingers = np.where(
    (cf_diagnosis == 1) & (age_months >= 18),
    np.random.binomial(1, 0.30, n_samples),
    0
)

# Step 8: Growth & Nutrition
# Weight percentile: CF shows decline after 2–6 months
weight_percentile = np.zeros(n_samples)
for i in range(n_samples):
    if cf_diagnosis[i] == 1:
        if age_months[i] <= 2:
            weight_percentile[i] = np.random.beta(2, 2) * 100  # normal at birth
        else:
            # Progressive falloff
            decay = min(1.0, (age_months[i] - 2) / 10)
            base = np.random.beta(1.5, 3) * 100
            weight_percentile[i] = max(1, base * (1 - decay))
    else:
        weight_percentile[i] = np.random.beta(2, 2) * 100

height_percentile = np.clip(weight_percentile + np.random.normal(0, 10, n_samples), 1, 99)
weight_for_height = (weight_percentile - height_percentile) / 10  # z-score approx

# Growth faltering: drop ≥2 channels (e.g., from 50th to <3rd)
growth_faltering = ((cf_diagnosis == 1) & (weight_percentile < 10)).astype(int)

# Appetite
appetite = np.where(
    cf_diagnosis == 1,
    np.random.choice([2, 3], p=[0.3, 0.7], size=n_samples),  # voracious despite poor gain
    np.random.choice([0, 1, 2], p=[0.05, 0.15, 0.80], size=n_samples)
)

# Stool
stool_character = np.empty(n_samples, dtype='U15')
for i in range(n_samples):
    if cf_diagnosis[i] == 1:
        stool_character[i] = np.random.choice(['greasy/oily', 'bulky', 'foul-smelling'], p=[0.5, 0.3, 0.2])
    else:
        stool_character[i] = np.random.choice(['normal', 'loose'], p=[0.9, 0.1])

stool_frequency = np.where(
    cf_diagnosis == 1,
    np.random.randint(3, 8, n_samples),
    np.random.randint(1, 4, n_samples)
)

# Step 9: GI Symptoms
abdominal_distention = np.where(cf_diagnosis == 1, np.random.binomial(1, 0.40, n_samples), np.random.binomial(1, 0.05, n_samples))
diarrhea_chronic = np.where(cf_diagnosis == 1, np.random.binomial(1, 0.35, n_samples), np.random.binomial(1, 0.08, n_samples))

# Fat malabsorption composite (0–3)
fat_malabsorption_signs = (
    (stool_character != 'normal').astype(int) +
    (growth_faltering) +
    (weight_for_height < -1).astype(int)
).clip(0, 3)

# Step 10: Infant-Specific Features (only relevant if age ≤ 12)
meconium_ileus = np.where(
    (cf_diagnosis == 1) & (age_months <= 2),
    np.random.binomial(1, 0.17, n_samples),  # 10–20% of CF cases [[11]]
    0
)

prolonged_jaundice = np.where(
    age_months <= 3,
    np.where(cf_diagnosis == 1, np.random.binomial(1, 0.20, n_samples), np.random.binomial(1, 0.05, n_samples)),
    0
)

failure_to_thrive = (weight_percentile < 3).astype(int)

# Step 11: Composite Scores
respiratory_score = (
    cough_type +
    (respiratory_infections_frequency / 2).clip(0, 3) +
    wheezing_present * 2 +
    nasal_polyps
).clip(0, 10)

nutritional_risk_score = (
    growth_faltering * 3 +
    (stool_character != 'normal').astype(int) * 2 +
    (appetite == 3).astype(int) +
    fat_malabsorption_signs
).clip(0, 10)

cf_clinical_suspicion_index = (
    salty_skin * 15 +
    family_history_cf * 10 +
    meconium_ileus * 20 +
    respiratory_score * 3 +
    nutritional_risk_score * 4 +
    (sweat_test_simulated > 60) * 25
).clip(0, 100)

# Step 12: Diagnostic Confidence & Age at Diagnosis
diagnostic_confidence = np.where(
    cf_diagnosis == 1,
    np.where(cf_clinical_suspicion_index >= 70, 3, np.where(cf_clinical_suspicion_index >= 40, 2, 1)),
    1
)

age_at_diagnosis = np.where(
    cf_diagnosis == 1,
    np.clip(age_months + np.random.randint(0, 3), 0, 24),
    np.nan
)

# Step 13: Assemble DataFrame
df = pd.DataFrame({
    'age_months': age_months,
    'family_history_cf': family_history_cf,
    'parental_carrier_status': parental_carrier_status,
    'ethnicity': ethnicity,
    'newborn_screening_result': newborn_screening_result,
    'salty_skin': salty_skin,
    'cough_type': cough_type,
    'cough_character': cough_character,
    'respiratory_infections_frequency': respiratory_infections_frequency,
    'wheezing_present': wheezing_present,
    'nasal_polyps': nasal_polyps,
    'clubbing_fingers': clubbing_fingers,
    'weight_percentile': weight_percentile.round(1),
    'height_percentile': height_percentile.round(1),
    'weight_for_height': weight_for_height.round(2),
    'growth_faltering': growth_faltering,
    'appetite': appetite,
    'stool_character': stool_character,
    'stool_frequency': stool_frequency,
    'abdominal_distention': abdominal_distention,
    'diarrhea_chronic': diarrhea_chronic,
    'fat_malabsorption_signs': fat_malabsorption_signs,
    'meconium_ileus': meconium_ileus,
    'prolonged_jaundice': prolonged_jaundice,
    'failure_to_thrive': failure_to_thrive,
    'sweat_test_simulated': sweat_test_simulated.round(1),
    'respiratory_score': respiratory_score.astype(int),
    'nutritional_risk_score': nutritional_risk_score.astype(int),
    'cf_clinical_suspicion_index': cf_clinical_suspicion_index.astype(int),
    'cf_diagnosis': cf_diagnosis,
    'diagnostic_confidence': diagnostic_confidence,
    'age_at_diagnosis': age_at_diagnosis
})

# Save to CSV
df.to_csv('synthetic_cystic_fibrosis_dataset.csv', index=False)

print(f"✅ Dataset generated with {len(df)} records.")
print(f"📊 CF cases: {df['cf_diagnosis'].sum()} ({df['cf_diagnosis'].mean()*100:.2f}%)")
print("📁 Saved as 'synthetic_cystic_fibrosis_dataset.csv'")


