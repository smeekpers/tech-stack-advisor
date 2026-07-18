import random
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Generate synthetic dataset
# -----------------------------

project_types = [
    "Web App",
    "API",
    "ML App",
    "Real-time App",
    "Mobile Backend",
]

perf_needs = ["Low", "Medium", "High"]

experiences = [
    "Beginner",
    "Intermediate",
    "Expert"
]

stacks = {
    "Web App": [
        "Django + SQLite",
        "Flask + SQLite",
        "Django + PostgreSQL",
        "Next.js + PostgreSQL",
    ],
    "API": [
        "Flask + SQLite",
        "FastAPI + SQLite",
        "FastAPI + PostgreSQL",
        "FastAPI + Redis",
    ],
    "ML App": [
        "FastAPI + TensorFlow",
        "FastAPI + PyTorch",
        "FastAPI + MLflow",
    ],
    "Real-time App": [
        "Node.js + Redis",
        "Node.js + Kafka",
        "Go + Redis",
        "Go + Kafka",
    ],
    "Mobile Backend": [
        "FastAPI + PostgreSQL",
        "Node.js + PostgreSQL",
        "Go + PostgreSQL",
    ]
}

rows = []

for _ in range(1000):

    project = random.choice(project_types)

    rows.append({
        "project_type": project,
        "team_size": random.randint(1, 20),
        "perf_need": random.choice(perf_needs),
        "experience": random.choice(experiences),
        "stack": random.choice(stacks[project])
    })

df = pd.DataFrame(rows)

print(f"Generated {len(df)} training samples")

# -----------------------------
# Encode categorical variables
# -----------------------------

encoders = {}

for col in [
    "project_type",
    "perf_need",
    "experience",
    "stack"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Train model
# -----------------------------

X = df[
    [
        "project_type",
        "team_size",
        "perf_need",
        "experience"
    ]
]

y = df["stack"]

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Save artifacts
# -----------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model saved to model.pkl")
print("Encoders saved to encoders.pkl")
