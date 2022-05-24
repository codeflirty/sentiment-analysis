# ========================================================================
# ------------------------------ VISUALIZATION ---------------------------
# ========================================================================

# Step 1:
# ================
# Imoprt Libraries
# ================

import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.pyplot import figure

# Step 2:
# =========
# Load Data
# =========

with open('Supporting Material/Output.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('Supporting Material/Output.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('Anger', 'Love', 'Fear', 'Happiness', 'Sadness', 'Surprise'))
        writer.writerows(lines)

data = pd.read_csv("Supporting Material/Output.csv")
data = data[-50:]

# Step 3:
# =============
# Visualization
# =============

color = ['red', 'green', 'blue', 'orange', 'black','brown']

def buildmebarchart(i=int):
    plt.legend(data.columns)
    p = plt.plot(data[:i].index, data[:i].values)
    for i in range(0,6):
        p[i].set_color(color[i])

fig = plt.figure(figsize=(12, 6), dpi=80)

plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
plt.subplots_adjust(bottom = 0.1, top = 0.9)
plt.title('Visualization')
plt.ylabel('Sentiment')
plt.xlabel('Extracted Tweet')

animator = ani.FuncAnimation(fig, buildmebarchart, interval = 100)
plt.show()

# ========================================================================
# ---------------------------- THANK YOU SO MUCH -------------------------
# ========================================================================