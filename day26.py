import sqlite3
import pandas as pd

# Synthetic educational healthcare data
patients = pd.DataFrame({
    'patient_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'department': [
        'Cardiology', 'Cardiology', 'Neurology', 'Neurology',
        'Cardiology', 'Neurology',  'Cardiology', 'Neurology'
    ],
    'risk_level': ['Low', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'Medium'],
    'cost': [1200, 5400, 1800, 7200, 2800, 4900, 900, 3200]
})

# Create a temporary database in memory
connection = sqlite3.connect(':memory:')

# Send the Pandas table into SQLite
patients.to_sql('patients', connection, index=False, if_exists='replace')

# SQL query: average cost by department
query = '''
SELECT department, COUNT(*) AS qualifying_patients, MAX(cost) AS highest_cost
FROM patients
WHERE risk_level = 'High' 
 AND cost > 3000
GROUP BY department; 
'''

result = pd.read_sql_query(query, connection)

print('AVERAGE COST BY DEPARTMENT')
print(result)

connection.close()