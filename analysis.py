import pandas as pd

# =========================
# LOAD FILES
# =========================
census = pd.read_csv("DDW-0000C-13.csv")  # age wise population
enrol = pd.read_csv("aadhaar_enrolment_state_cleaned.csv")  # Aadhaar enrolments
demo = pd.read_csv("aadhaar_demographic_state_cleaned.csv")  # Aadhaar demographic records

# =========================
# STEP 1: PREPARE CENSUS DATA
# =========================

# Keep only India rows (if repeated by age)
census = census[census['state'] == "India"]

# Ensure Age is numeric so comparisons work
census['Age'] = pd.to_numeric(census['Age'], errors='coerce')
census = census.dropna(subset=['Age'])

# Aggregate Urban & Rural by age group
def age_range(df, low, high):
    return df[(df['Age'] >= low) & (df['Age'] <= high)]

total_pop = census['Total Persons'].sum()

urban_8_60 = age_range(census, 8, 60)['Urban Persons'].sum()
urban_8_17 = age_range(census, 8, 17)['Urban Persons'].sum()
urban_18_60 = age_range(census, 18, 60)['Urban Persons'].sum()

rural_18_60 = age_range(census, 18, 60)['Rural Persons'].sum()

# =========================
# STEP 2: MERGE AADHAAR DATA BY STATE + DISTRICT
# =========================

# Enrolment Aadhaar (use state + district)
enrol_group = enrol.groupby(['state','district'], as_index=False)[['age_5_17','age_18_greater']].sum()
enrol_group['aadhaar_enrolments'] = enrol_group['age_5_17'] + enrol_group['age_18_greater']

# Demographic Aadhaar (use state + district)
demo_group = demo.groupby(['state','district'], as_index=False)[['demo_age_5_17','demo_age_17_']].sum()
demo_group['aadhaar_records'] = demo_group['demo_age_5_17'] + demo_group['demo_age_17_']

# Merge both
aadhaar = pd.merge(enrol_group, demo_group, on=['state','district'], how="outer")

# Fill missing enrol/record counts with 0 so PDI can be computed
aadhaar[['aadhaar_enrolments','aadhaar_records']] = aadhaar[['aadhaar_enrolments','aadhaar_records']].fillna(0)

# =========================
# STEP 3: COMPUTE COMPONENTS (Normalized)
# =========================

aadhaar['W'] = urban_18_60 / total_pop
aadhaar['S'] = urban_8_17 / total_pop
aadhaar['R'] = rural_18_60 / total_pop

aadhaar['A'] = aadhaar['aadhaar_enrolments'] / total_pop
aadhaar['D'] = aadhaar['aadhaar_records'] / total_pop

# =========================
# STEP 4: FINAL TELECOM PDI
# =========================

aadhaar['PDI'] = (
    0.30 * aadhaar['W'] +
    0.15 * aadhaar['S'] +
    0.10 * aadhaar['R'] +
    0.25 * aadhaar['A'] +
    0.20 * aadhaar['D']
)

# =========================
# STEP 5: SORT HOTSPOTS
# =========================

# Select relevant columns
result = aadhaar[['state','district','PDI']]

# Remove NaN PDI rows (optional)
result = result.dropna(subset=['PDI'])

# Sort descending
result = result.sort_values(by='PDI', ascending=False)

# Print top 20
top20 = result.head(20)
print("Top 20 Telecom Hotspot Districts:")
print(top20)


#to print all data of telecom  
#result = aadhaar[['state','district','PDI']].sort_values(by='PDI', ascending=False)
#print(result)



# =========================
# STEP 6: EDUCATION PDI
# =========================

urban_3_18 = age_range(census, 3, 18)['Urban Persons'].sum()
rural_3_18 = age_range(census, 3, 18)['Rural Persons'].sum()

aadhaar['U_s'] = urban_3_18 / total_pop
aadhaar['R_s'] = rural_3_18 / total_pop

# Aadhaar 0–17 enrolments already inside age_5_17 + ??? (you must adjust based on file columns)
aadhaar['A_es'] = aadhaar['age_5_17'] / total_pop

# Aadhaar biometric updates 5–17 from demo file
aadhaar['A_bs'] = aadhaar['demo_age_5_17'] / total_pop

aadhaar['Education_PDI'] = (
    0.50 * aadhaar['U_s'] +
    0.25 * aadhaar['R_s'] +
    0.15 * aadhaar['A_es'] +
    0.10 * aadhaar['A_bs']
)


# =========================
# STEP 7: TRANSPORT PDI
# =========================

urban_m_15_60 = age_range(census, 15, 60)['Urban Males'].sum()
urban_f_15_60 = age_range(census, 15, 60)['Urban Females'].sum()
rural_w_15_60 = age_range(census, 15, 60)['Rural Persons'].sum()

aadhaar['U_m'] = urban_m_15_60 / total_pop
aadhaar['U_f'] = urban_f_15_60 / total_pop
aadhaar['R_w'] = rural_w_15_60 / total_pop

aadhaar['A_e'] = aadhaar['aadhaar_enrolments'] / total_pop
aadhaar['A_d'] = aadhaar['aadhaar_records'] / total_pop

aadhaar['Transport_PDI'] = (
    0.40 * aadhaar['U_m'] +
    0.25 * aadhaar['U_f'] +
    0.05 * aadhaar['R_w'] +
    0.10 * aadhaar['A_e'] +
    0.10 * aadhaar['A_d']
)



# =========================
# STEP 8: UTILITIES PDI
# =========================

aadhaar['P_t'] = total_pop / total_pop  # always 1 normalized

aadhaar['A_eu'] = aadhaar['aadhaar_enrolments'] / total_pop
aadhaar['A_du'] = aadhaar['aadhaar_records'] / total_pop

aadhaar['Utilities_PDI'] = (
    0.70 * aadhaar['P_t'] +
    0.15 * aadhaar['A_eu'] +
    0.15 * aadhaar['A_du']
)


# =========================
# STEP 9: JOBS PDI
# =========================

workforce_21_30 = age_range(census, 21, 30)['Total Persons'].sum()
aadhaar['W_21_30'] = workforce_21_30 / total_pop

aadhaar['A_ej'] = aadhaar['aadhaar_enrolments'] / total_pop
aadhaar['A_bj'] = aadhaar['aadhaar_records'] / total_pop

aadhaar['Jobs_PDI'] = (
    0.60 * aadhaar['W_21_30'] +
    0.20 * aadhaar['A_ej'] +
    0.20 * aadhaar['A_bj']
)


# =========================
# STEP 10: NPI
# =========================

aadhaar['NPI'] = (
    aadhaar['PDI'] +
    aadhaar['Transport_PDI'] +
    aadhaar['Education_PDI'] +
    aadhaar['Utilities_PDI'] +
    aadhaar['Jobs_PDI']
) / 5

print("\nTOP-20 Telecom:")
print(aadhaar[['state','district','PDI']].sort_values(by='PDI', ascending=False).head(20))

print("\nTOP-20 Education:")
print(aadhaar[['state','district','Education_PDI']].sort_values(by='Education_PDI', ascending=False).head(20))

print("\nTOP-20 Transport:")
print(aadhaar[['state','district','Transport_PDI']].sort_values(by='Transport_PDI', ascending=False).head(20))

print("\nTOP-20 Utilities:")
print(aadhaar[['state','district','Utilities_PDI']].sort_values(by='Utilities_PDI', ascending=False).head(20))

print("\nTOP-20 Jobs:")
print(aadhaar[['state','district','Jobs_PDI']].sort_values(by='Jobs_PDI', ascending=False).head(20))

print("\nTOP-20 NPI (FINAL NATIONAL PRIORITY):")
print(aadhaar[['state','district','NPI']].sort_values(by='NPI', ascending=False).head(20))



#case study to get the top states and their top districts for each PDI 
# =========================
# PHASE-2: TOP STATES + TOP DISTRICTS
# =========================

pdi_list = ['PDI', 'Education_PDI', 'Transport_PDI', 'Utilities_PDI', 'Jobs_PDI', 'NPI']

for pdi in pdi_list:
    print(f"\n====== {pdi} ======")

    # --- TOP 10 STATES by mean district PDI ---
    top_states = (
        aadhaar.groupby('state', as_index=False)[pdi]
        .mean()
        .sort_values(by=pdi, ascending=False)
        .head(10)
    )
    print("\nTop 10 States:")
    print(top_states)

    # --- TOP 5 DISTRICTS per top state ---
    print("\nTop 5 Districts of Each Top State:")

    for st in top_states['state']:
        top_districts = (
            aadhaar[aadhaar['state'] == st][['state','district', pdi]]
            .sort_values(by=pdi, ascending=False)
            .head(5)
        )
        print(f"\n>> {st} — Top Districts:")
        print(top_districts)