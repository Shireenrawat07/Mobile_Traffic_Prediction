import matplotlib.pyplot as plt
import pandas as pd

file = "Dataset/full_dataset.csv"
df = pd.read_csv(file)
# Debug print to check model output
print("\n================ DEBUG INFO ================")
print("Sample predicted values:", preds[:10])
print("Sample actual values:", actual[:10])
print("Mean of predictions:", preds.mean())
print("Standard deviation of predictions:", preds.std())
print("============================================\n")


plt.figure(figsize=(10, 5))
plt.plot(df.iloc[:, 0].values, label="Traffic Data")
plt.title("Traffic Volume Over Time")
plt.xlabel("Time")
plt.ylabel("Traffic Value")
plt.legend()
plt.show()
