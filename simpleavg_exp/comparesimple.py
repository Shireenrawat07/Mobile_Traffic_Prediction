import pandas as pd
import matplotlib.pyplot as plt

# Load with headers (do NOT use names=[])
fed = pd.read_csv("simpleavg_exp/fedavg_results.csv")
median = pd.read_csv("simpleavg_exp/simpleavg_results.csv")

# Print to verify column names
print(fed)
print(median)

# Plot
plt.plot(fed["Round"], fed["Loss"], label="FedAvg Loss", linewidth=2)
plt.plot(median["Round"], median["Loss"], label="SimpleAvg Loss", linewidth=2)

plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("FedAvg vs Simple Loss")
plt.legend()
plt.grid(True)
plt.savefig("plots/simple_fed_comp.png", dpi=300, bbox_inches="tight")
plt.show()  # normal scale
