import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=" * 60)
print ("Day 22: MERGING MULTIPLE PATIENT DATASETS")
print("=" * 60)

# =================================================================
# 1. LOAD ALL THREE DATASETS
# =================================================================

demo = pd.read_csv('demographics.csv')
labs = pd.read_csv('lab_results.csv')
images = pd.read_csv('imaging.csv.py')

print("\n=== Demographics (EHR) ---")
print(f"Records: {len(demo)} | IDs: {sorted(demo['patient_id'].tolist())}")
print(demo[['patient_id', 'name', 'age', 'adhd_diagnosis']])

print("\n--- Lab Results (LIS) ---")
print(f"Records: {len(images)} | IDs: {sorted(images['patient_id'].tolist())}")
print(labs[['patient_id', 'wbc', 'hgb', 'fasting_glucose', 'crp']])

print("\n--- Imaging (PACS) ---")
print(f"records: {len(images)} | IDs: {sorted(images['patient_id'].tolist())}")
print(images[['patient_id', 'amygdala_vol_left', 'amygdala_vol_right', 'radiologist_finding']])

# =========================================================
# 2. THE FOUR MERGE TYPES
# =========================================================

# The syntax is always:
# table1.merge(table2,on='common_column', how='type')
#
# 'how' can be: inner, left, right, outer

# _____ 2a. INNER JOIN ______ keeps only patients in All datasets
#       "I ony want patients with complete data for my ML model"
inner = demo.merge(labs,on='patient_id', how='inner') \
            .merge(images, on='patient_id', how='inner')

print(f"\n INNER JOIN: {len(inner)} patients (complete data only)")
print(f" Kept: {sorted(inner['patient_id'].tolist())}")
print(f" dropped: {len(demo) - len(inner)} patients")
print(f" Use case: Ml training - no missing values")

# 2b. LEFT JOIN __ keeps All patientd from the left table (demo)
#   "Show me every patient in the EHR, fill in labs/images if available"
left = demo.merge(labs, on='patient_id', how='left') \
           .merge(images, on='patient_id', how='left')

print(f"\n LEFT JOIN: {len(left)} patients (all EHR patients)")
print(f"  Missing labs: {left['wbc'].isna().sum} patient(s)")
print(f"  Missing imaging: {left['amygdala_vol_left'].isna().sum()} patient(s)")
print(f"  Usse case: Clinical audit - flag who has gaps")

# 2c. OUTER JOIN: Keeps EVERY psatient from EVERY dataset
#  "Show me everything - i neeed to go find orphan records"
outer = demo.merge(labs, on='patient_id', how='outer') \
            .merge(images, on='patient_id', how='outer')

orphans_labs = set(labs['patient_id']) - set(demo['patient_id'])
orphans_img  = set(images['patient_id']) - set(demo['patient_id'])

print(f"n OUTER JON: {len(outer)} patients (everything)")
print(f"  Orphans in lab only: {orphans_labs}")
print(f"  Orphans in imaging only: {orphans_img}")
print(f" Use case: Data quality audit")

#  2d. ANTI JOIN : patients in demographics but NOT in imaging
#  "Which patients still need an MRI ordered?"
no_imaging = demo[~demo['patient_id'].isin(images['patient_id'])]

print(f"\n ANTI JOIN; {len(no_imaging)} patients WITHOUT imaging")
for _, row in no_imaging.iterrows():
    print(f"  Patient {row['patient_id']} - {row['name']} (age {row['age']}) - needs MRI ordered")
print(f"  Use case: Operations - radiology follow-up list")

# ===================================================================
# 3. DERIVED CLINICAL FEATURES (on the LEFT join)
# ===================================================================
# Once data is merged, you can create new columns that combine information from multiple systems. This is where clinical reasoning meets data science.

df = left.copy()

# Anemia risk: different hemoglobin thresholds for M vs F
df['anemia_risk'] = np.where(
    (df['gender'] == 'F') & (df['hgb'] < 12.0), 'High',
    np.where((df['gender'] == 'M') & (df['hgb'] < 13.5), 'High', 'Normal')
)

# Diabetes flag; based on fasting glucose thresholds
df['diabetes_flag'] = np.where(
    df['fasting_glucose'] >= 126, 'Diabetic',
    np.where(df['fasting_glucose'] >= 100, 'Prediabetic', 'Normal')
)

# Inflammation flag: CRP > 5 is clinically elevated
df['inflammation_flag'] = np.where(
    df['crp'] > 5.0, 'Elevated CRP', 'Normal'
)

# Total amygdala volume (left + right) - a key ADHD research metric
df['amygdala_total'] = df['amygdala_vol_left'] + df['amygdala_vol_right']

print("\n" + "=" * 60)
print(" MERGED DATASET WITH DERIVED FEATURES")
print("=" * 60)
cols = ['patient_id', 'name', 'age', 'adhd_diagnosis',
        'fasting_glucose', 'diabetes_flag', 'crp', 'inflammation_flag',
        'amygdala_total', 'radiologist_finding']
print(df[cols].to_string(index=False))

print(f"\nMissing data summary:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# =======================================================
# 4. CLINICAL INSIGHTS
# =======================================================

print("\n" + "=" * 60)
print(" CLINICAL INSIGHTS FROM MERGED DATA")
print("=" * 60)

# Insight 1: Diabetes breakdown
diabetic = df[df['diabetes_flag'] == 'Diabetic']
prediabetic = df[df['diabetes_flag'] == 'Prediabetic']
print(f"\n Diabetes: {len(diabetic)} diabetic, {len(prediabetic)} prediabetic")
if len(diabetic) > 0:
    for _, row in diabetic.iterrows():
        print(f"   . {row['name']} - glucose: {row['fasting_glucose']:.0f} mg/dl")

# Insight 2: ADHD vs Amygdala volume
adhd = df[(df['adhd_diagnosis'] == 'Yes') & df['amygdala_total'].notna()]
control  = df[(df['adhd_diagnosis'] == 'No') & df['amygdala_total'].notna()]

if len(adhd) > 0 and len(control) > 0:
    adhd_mean = adhd['amygdala_total'].mean()
    ctrl_mean = control['amygdala_total'].mean()
    diff_pct = ((ctrl_mean - adhd_mean) / ctrl_mean) * 100
    print(f"\n Amygdala Volume:")
    print(f"  ADHD mean:  {adhd_mean:.2f} cm")
    print(f"  Control mean:  {ctrl_mean:.2f} cm")
    print(f"  ADHD is {diff_pct:.1f}% smaller - aligns with ENIGMA findings")

# Insight 3: Elevated inflammation + imaging correlation
elevated = df[df['inflammation_flag'] == 'Elevated CRP']
if len(elevated) > 0:
    print(f"\n Elevated CRP Patients:")
    for _, row in elevated.iterrows():
        finding = row['radiologist_finding'] if pd.notna(row['radiologist_finding']) else 'No imaging'
        print(f"  . {row['name']} - CRP: {row['crp']:.1f} | Imaging; {finding}")

# =================================================================================================
# 5. VISUALIZATION
# =================================================================================================

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

fig.suptitle('Day 22: Multi-Source Patient Data Integration',
             fontsize=16, fontweight='bold', color='#2c3e50')

# Plot 1: Data completeness per patient
ax1 = fig.add_subplot(gs[0, 0])
completeness = pd.DataFrame({
    'Patient': [f'P{p}' for p in df['patient_id']],
    'Demographics': [1] * len(df),
    'Lab Results': df['wbc'].notna().astype(int),
    'Imaging': df['amygdala_vol_left'].notna().astype(int)
    })
completeness.set_index('Patient', inplace=True)
completeness.plot(kind='barh', stacked=True, ax=ax1,
                  color=['#2c3e50', '#3498db', '#27ae60'], width=0.7)
ax1.set_title('Data completeness per Patient', fontweight='bold')
ax1.set_xlim(0, 3.5)
ax1.legend(loc='lower right', fontsize=7)

# Plot 2: Records per sytsems
ax2 = fig.add_subplot(gs[0, 1])
systems = ['Demographics\n(EHR', 'Lab Results\n(LTS)', 'Imaging\n(PACS)']
counts = [len(demo), len(labs), len(images)]
bars = ax2.bar(systems, counts, color=['#2c3e50', '#3498db', '#27ae60'],
               edgecolor='black')
for bar, c in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(c), ha='center', fontweight='bold', fontsize=12)
ax2.set_title('Records per Sytsems', fontweight='bold', fontsize=12)

# Plot 3: merge type comparison
ax3 = fig.add_subplot(gs[0, 2])
merge_names = ['INNER\n(Complete\nOnly)', 'LEFT\n(ALL EHR\nPatients)', 'OUTER\n(All\nRecords)']
merge_counts = [len(inner), len(left), len(outer)]
bars = ax3.bar(merge_names, merge_counts,
               color=['#27ae60', '#3498db', '#f39c12'], edgecolor='black')
ax3.set_title('Patienta per Merge Stragedy', fontweight='bold')
for bar, c in zip(bars, merge_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(c), ha='center', fontweight='bold', fontsize=12)

# plot 4: Amygdala volume - ADHD vs Control
ax4 = fig.add_subplot(gs[1, 0])
valid = df[df['amygdala_total'].notna()]
for _, row in valid.iterrows():
    color = '#e74c3c' if row['adhd_diagnosis'] == 'Yes' else '#3498db'
    marker = 's' if row['adhd_diagnosis'] == 'Yes' else 'o'
    ax4.scatter(row['age'], row['amygdala_total'], c=color, marker=marker,
                s=150, edgecolors='black', linewidth=1.2, zorder=3)
ax4.scatter([], [], c='#e74c3c', marker='s', s=80, edgecolors='black', label='ADHD')
ax4.scatter([], [], c='#3498db', marker='o', s=80, edgecolors='black', label='Control')
ax4.legend()
ax4.set_xlabel('Age (years)')
ax4.set_ylabel('Total Amygdala Volume (cm)')
ax4.set_title('Amygdala Volume: ADHD vs Control', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Glucose by diabetes status
ax5 = fig.add_subplot(gs[1, 1])
status_colors = {'Normal': '#27ae60', 'Prediabetic': '#f39c12', 'Diabetic': '#e74c3c'}
for i, status in enumerate(['Normal', 'Prediabetic', 'Diabetic']):
    group = df[df['diabetes_flag'] == status]
    x_jitter = np.random.uniform(-0.12, 0.12, len(group))
    ax5.scatter([i] * len(group) + x_jitter, group['fasting_glucose'],
                c=status_colors[status], s=100, edgecolors='black',
                linewidth=0.8, zorder=3)
ax5.axhline(y=100, color='#f39c12', linestyle='--', linewidth=1.5, label='Prediabetes (100)')
ax5.axhline(y=126, color='#e74c3c', linestyle='--', linewidth=1.5, label='Diabetes (126)')
ax5.set_xticks([0, 1, 2])
ax5.set_xticklabels(['Normal', 'Prediabetic', 'Diabetic'])
ax5.set_ylabel('Fasting Glocuse (mg/dl)')
ax5.set_title('Glucose by Diabetes Status', fontweight='bold')
ax5.legend(fontsize=7)

# Plot 6: Summary panel
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary = (
    f"MERGE SUMMARY\n"
    f"{"-" * 22}\n"
    f"EHR patients:         {len(demo)}\n"
    f"With lab results:     {df['wbc'].notna().sum}/{len(df)}\n"
    f"With imaging:         {df['amygdala_vol_left'].notna().sum()}/{len(df)}\n"
    f"Complete (inner):     {len(inner)}\n"
    f"\n"
    f"FINDINGS\n"
    f"{'-' * 22}\n"
    f"Diabetic:             {len(diabetic)}\n"
    f"Elevated CRP:         {len(prediabetic)}\n"
    f"ADHD patients:        {len(adhd)}\n"
    f"Needs imaging:        {len(no_imaging)}\n"
    f"\n"
    f"AMYGDALA (ADHD vs Ctrl)\n"
    f"{'-' * 22}\n"
    f"ADHD mean:            {adhd_mean:.2f} cm\n"
    f"Control:              {ctrl_mean:.2f} cm\n"
    f"Diff:                 {ctrl_mean - adhd_mean:.2f} cm"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

plt.savefig("day22_merge_analysis.png", dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
print("\n" + "=" * 60)
print(" Dashboard saved: day22_merge_analysis.png")
print(" Day 22 complete!")
print("=" * 60)


