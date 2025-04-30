# 4) Plot class distribution
counts = df["class"].value_counts().sort_values()
plt.figure(figsize=(8,4))
counts.plot.barh()
plt.title("UrbanSound8K Class Counts")
plt.xlabel("Number of examples")
plt.tight_layout()
plt.show()



# Pie chart of class distribution

plt.figure(figsize=(6, 6))
df["class"].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title("UrbanSound8K Class Distribution")
plt.ylabel("")  # remove y-axis label
plt.tight_layout()
plt.show()
