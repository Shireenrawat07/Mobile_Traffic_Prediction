import os
import json
import pandas as pd

# Folders for each algorithm
folders = {
    'FedAvg': 'fedavg_results',
    'FedProx': 'fedprox_results'
}

rows = []

for algo, folder in folders.items():
    for file in os.listdir(folder):
        if file.endswith('.json') and file.startswith('metrics_alpha_'):
            # Extract alpha value from filename
            alpha = float(file.replace('metrics_alpha_','').replace('.json',''))
            
            # Load JSON
            with open(os.path.join(folder, file), 'r') as f:
                data = json.load(f)
            
            # data should have client-wise metrics
            # Assuming structure: { "client1": {"MAE":0.5, "RMSE":0.65, "NRMSE":0.12}, ... }
            for client_key, metrics in data.items():
                 
                client_num = int(client_key.split('_')[-1]) # convert "client1" -> 1
                rows.append({
                    'Client': client_num,
                    'Split': alpha,
                    'Algorithm': algo,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'NRMSE': metrics['NRMSE']
                })

# Convert to DataFrame and save CSV
df_results = pd.DataFrame(rows)
df_results.to_csv('results_fedprox_fedavg.csv', index=False)
print("CSV file 'results_fedprox_fedavg.csv' created successfully!")