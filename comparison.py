import json
import matplotlib.pyplot as plt

# Load centralized data
with open("Loss/centralized_loss.json") as f:
    centralized_data = json.load(f)

# Load hierarchical data
with open("Loss/hierarchical_loss.json") as f:
    hierarchical_data = json.load(f)

# Extract values
centralized_loss = centralized_data["loss_history"]
centralized_time = centralized_data["avg_coord_time"]

hierarchical_loss = hierarchical_data["loss_history"]
hierarchical_time = hierarchical_data["avg_coord_time"]

# -------------------------
# Convergence Plot
# -------------------------

plt.plot(centralized_loss, label="Centralized")
plt.plot(hierarchical_loss, label="Hierarchical")

plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Centralized vs Hierarchical Convergence")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Coordination Time Plot
# -------------------------

labels = ["Centralized", "Hierarchical"]
values = [centralized_time, hierarchical_time]

plt.bar(labels, values)
plt.ylabel("Average Coordination Time (s)")
plt.title("Coordination Overhead Comparison")
plt.grid(axis="y")
plt.show()