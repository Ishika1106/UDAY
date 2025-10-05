import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n = 4000

# Generate class values
classes = [f"Class {i}" for i in range(6, 13)]
class_choices = np.random.choice(classes, size=n)

# Generate random names (simple pattern)
names = [f"Student_{i+1}" for i in range(n)]

# Attendance and marks
attendance = np.clip(np.random.normal(75, 18, n), 0, 100).round(1)
marks = np.clip((attendance * 0.6) + np.random.normal(20, 15, n), 0, 100).round(1)

# Fees (1 = Unpaid, 0 = Paid)
prob_unpaid = 0.12 + (np.maximum(0, 50 - attendance) / 500) + (np.maximum(0, 45 - marks) / 500)
fees = np.where(np.random.rand(n) < prob_unpaid, 1, 0)

# Dropout risk (1 = High, 0 = Low/Medium)
risk_score = (100 - attendance) * 0.5 + (100 - marks) * 0.35 + (fees * 12) + np.random.normal(0, 6, n)
risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100
droupoutrisk = np.where(risk_score > 65, 1, 0)

# Create DataFrame with proper column names
df = pd.DataFrame({
    "Name": names,
    "Class": class_choices,
    "Attendance": attendance,
    "Fees": fees,
    "Marks": marks,
    "DropoutRisk": droupoutrisk
})

# Save to Excel
df.to_excel("dropout_reference_numeric.xlsx", index=False)
print("✅ File saved as dropout_reference_numeric.xlsx")
