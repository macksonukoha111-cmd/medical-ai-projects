import pandas as pd
# Synthetic educational healthcare data
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6, 6, 6],
    'department': ['Cardiology', 'Neurology', 'Cardiology', 'Cardiology',
                   'Neurology', 'Cardiology', 'Neurology', 'Neurology'],
    'test-name': ['Troponin', 'MRI', 'Troponin', 'Troponin',
                  'MRI', 'CBC', 'CBC', 'CBC'],
    'cost': [1200, 5400, 1800, 1800, 7200, 900, 1400, 1400]                             
})

print('PATIENT RECORDS')
print(patients)

print('\nTOTAL ROWS')
print(len(patients))

print('\nDUPLICATE ROW FLAGS')
print(patients.duplicated())

print('\nDUPLICATE RECORDS')
duplicates = patients[patients.duplicated()]
print(duplicates)

print('\nRECORDS AFTER REMOVING DUPLICATES')
clean_patients = patients.drop_duplicates()
print(clean_patients)
print(f'Rows before cleaning: {len(patients)}')
print(f'Rows after cleaning: {len(clean_patients)}')
print(F'Duplicates removed: {len(patients) - len(clean_patients)}')