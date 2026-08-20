import sqlite3
import pandas as pd

# Synthetic educational healthcare data
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'department': ['Cardiology', 'Cardiology', 'Neurology', 'Neurology',
                   'Cardiology', 'Neurology', 'Cardiology', 'Neurology'],
    'risk_level': ['Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'Medium'],
    'cost': [1200, 5400, 1800, 7200, 2800, 4900, 900, 3200]              
})

connection = sqlite3.connect(':memory:')
patients.to_sql('patients', connection, index=False, if_exists='replace')

# SQL retrieves only high-risk records
query = '''
SELECT patient_id, department,
risk_level, cost
FROM patients
WHERE risk_level = 'High';
'''
high_risk = pd.read_sql_query(query, connection)
connection.close()

print('HIGH-RISK PATIENTS')
print(high_risk)

print('\nAVERAGE COST BY DEPARTMENT')
summary = high_risk.groupby('department') ['cost'].agg(['count', 'mean', 'max'])
print(summary)