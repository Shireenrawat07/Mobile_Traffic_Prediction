import os
import json
import pandas as pd

# Folders
folders = {
    'FedAvg': 'results/fedavg_results',
    'FedNova': 'results/fednova_results'
}

rows = []

for algo, folder in folders.items():
    for file in os.listdir(folder):
        if file.endswith('.json') and file.startswith('metrics_alpha_'):
            # Extract alpha
            alpha = float(file.replace('metrics_alpha_','').replace('.json',''))
            
            # Load JSON
            with open(os.path.join(folder, file), 'r') as f:
                data = json.load(f)
            
            # Loop over clients
            for client_key, metrics in data.items():
                client_num = int(client_key.split('_')[-1])  # works for client1 or client_1
                rows.append({
                    'Client': client_num,
                    'Split': alpha,
                    'Algorithm': algo,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'NRMSE': metrics['NRMSE']
                })

# Save CSV
df_results = pd.DataFrame(rows)
df_results.to_csv('results/results_fedavg_fednova.csv', index=False)
print("CSV file 'results_fedavg_fednova.csv' created successfully!")