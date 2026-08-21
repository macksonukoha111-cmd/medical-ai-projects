import pandas as pd

# synthetic educational healthcare data
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 3, 4, 5, 6, 6,],
    'age': [25, 40, None, None, 200, 62, 30, 30],
    'department': ['Cardiology', 'Neurology', 'Cardiology', 'Cardiology', 'Neurology','Cardiology', 'Neurology','Neurology'],
    'blood_pressure': [120, 135, 999, 999, 160, None, 130, 130],
    'cost': [1200, 5400, 1800, 1800, 7200, 900, 1400, 1400]
})

def create_quality_report(data): 
    report = {
        'total_rows': len(data),
        'total_columns':
len(data.columns),
        'duplicate_rows':
int(data.duplicated().sum()),
        'total_missing_values':
int(data.isnull().sum().sum()),
        'missing_patient_ids':
int(data['patient_id'].isnull().sum()),
         'invalid_age_records':
int((data['age'] > 120).sum()),
          'invalid_bp_records':
int((data['blood_pressure'] > 300).sum())                                       
    } 
    return report 

print('RAW DATA')
print(patients)

print('\nDATA-QUALITY REPORT')
quality_report = create_quality_report(patients)

for metrics, value in quality_report.items(): 
    print(f'{metrics}: {value}')