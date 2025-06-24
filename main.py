import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Upload 'student-mat.csv' and 'student-por.csv' before running this
math_df = pd.read_csv("student-mat.csv", sep=';')
port_df = pd.read_csv("student-por.csv", sep=';')
common_students = pd.merge(
    math_df,
    port_df,
    on=[
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
        "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
    ],
    suffixes=("_math", "_por")
)
df = math_df.copy()

label_encoders = {}
categorical_columns = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
X = df.drop(columns=["G3"])
y = df["G3"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, bins=15)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
importances = model.feature_importances_
features = df.drop(columns="G3").columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 20], [0, 20], 'r--')
plt.title("Actual vs Predicted Grades")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.grid(True)
plt.show()
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.show()
plt.figure(figsize=(16, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of All Features")
plt.show()
# We'll decode the 'studytime' to make it more readable (optional)
studytime_labels = {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}
df_box = math_df.copy()
df_box["studytime"] = df_box["studytime"].map(studytime_labels)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_box, x="studytime", y="G3", palette="Set2")
plt.title("Final Grades (G3) vs Study Time")
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Grade (G3)")
plt.grid(True)
plt.show()
plt.figure(figsize=(6, 4))
sns.boxplot(data=math_df, x="internet", y="G3", palette="pastel")
plt.title("G3 vs Internet Access at Home")
plt.xlabel("Internet Access")
plt.ylabel("Final Grade (G3)")
plt.grid(True)
plt.show()
plt.figure(figsize=(7, 5))
sns.boxplot(data=math_df, x="failures", y="G3", palette="coolwarm")
plt.title("G3 vs Number of Past Class Failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:50], label="Actual", marker='o')
plt.plot(y_pred[:50], label="Predicted", marker='x')
plt.title("Actual vs Predicted Grades (Sample of 50 Students)")
plt.xlabel("Student Index")
plt.ylabel("Final Grade (G3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
sample_input = X_test[0].reshape(1, -1)
predicted_g3 = model.predict(sample_input)
print(f"🎓 Predicted G3 for test input: {predicted_g3[0]:.2f}")


