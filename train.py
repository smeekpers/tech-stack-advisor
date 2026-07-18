import random
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Possible values
# --------------------------------------------------

project_types = [
    "Web App",
    "API",
    "ML App",
    "Real-time App",
    "Mobile Backend"
]

cloud_providers = [
    "AWS",
    "Azure",
    "GCP",
    "On-Prem"
]

budgets = [
    "Low",
    "Medium",
    "High",
    "Enterprise"
]

expected_users = [
    "Small",
    "Medium",
    "Large",
    "Massive"
]

deployment_targets = [
    "Docker",
    "Kubernetes",
    "VM",
    "Serverless"
]

database_preferences = [
    "SQLite",
    "PostgreSQL",
    "MongoDB",
    "Redis"
]

experiences = [
    "Beginner",
    "Intermediate",
    "Expert"
]

# --------------------------------------------------
# Generate synthetic training data
# --------------------------------------------------

rows = []

for _ in range(1000):

    project = random.choice(project_types)
    team_size = random.randint(1, 20)

    cloud = random.choice(cloud_providers)
    budget = random.choice(budgets)
    users = random.choice(expected_users)
    deployment = random.choice(deployment_targets)
    database = random.choice(database_preferences)
    experience = random.choice(experiences)

    # ----------------------------------------------
    # Recommendation rules
    # ----------------------------------------------

    if project == "ML App":

        if deployment == "Kubernetes":
            stack = "FastAPI + TensorFlow + Kubernetes"
        else:
            stack = random.choice([
                "FastAPI + TensorFlow",
                "FastAPI + PyTorch"
            ])

    elif project == "Real-time App":

        if users in ["Large", "Massive"]:
            stack = random.choice([
                "Node.js + Kafka",
                "Go + Kafka"
            ])
        else:
            stack = random.choice([
                "Node.js + Redis",
                "Go + Redis"
            ])

    elif project == "API":

        if budget in ["High", "Enterprise"]:
            stack = "FastAPI + PostgreSQL"
        else:
            stack = random.choice([
                "Flask + SQLite",
                "FastAPI + SQLite"
            ])

    elif project == "Web App":

        if budget == "Low":
            stack = "Flask + SQLite"

        elif deployment == "Kubernetes":
            stack = "Next.js + PostgreSQL"

        else:
            stack = "Django + PostgreSQL"

    else:  # Mobile Backend

        if cloud == "AWS":
            stack = "FastAPI + PostgreSQL"

        elif cloud == "Azure":
            stack = "FastAPI + SQL Server"

        else:
            stack = "Node.js + PostgreSQL"

    rows.append({
        "project_type": project,
        "team_size": team_size,
        "cloud_provider": cloud,
        "budget": budget,
        "expected_users": users,
        "deployment_target": deployment,
        "database_preference": database,
        "experience": experience,
        "stack": stack
    })

df = pd.DataFrame(rows)

print(f"Generated {len(df)} training samples")

# --------------------------------------------------
# Encode categorical features
# --------------------------------------------------

encoders = {}

for col in [
    "project_type",
    "cloud_provider",
    "budget",
    "expected_users",
    "deployment_target",
    "database_preference",
    "experience",
    "stack"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df[
    [
        "project_type",
        "team_size",
        "cloud_provider",
        "budget",
        "expected_users",
        "deployment_target",
        "database_preference",
        "experience"
    ]
]

y = df["stack"]

# --------------------------------------------------
# Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X, y)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = model.predict(X)

print(
    f"Training Accuracy: "
    f"{accuracy_score(y, predictions):.2f}"
)

# --------------------------------------------------
# Save model
# --------------------------------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ model.pkl saved")
print("✅ encoders.pkl saved")
