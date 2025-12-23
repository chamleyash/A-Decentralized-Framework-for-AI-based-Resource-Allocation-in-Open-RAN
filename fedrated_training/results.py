import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("metrics.csv")

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(data["Throughput"])
plt.title("Throughput")

plt.subplot(1,2,2)
plt.plot(data["Latency"])
plt.title("Latency")

plt.tight_layout()
plt.show()
