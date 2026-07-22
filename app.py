import gradio as gr
import pandas as pd
import pickle

# -------------------------------------
# Load model and encoders
# -------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# -------------------------------------
# Prediction function
# -------------------------------------

def recommend_stack(
    project_type,
    team_size,
    cloud_provider,
    budget,
    expected_users,
    deployment_target,
    database_preference,
    experience
):

    sample = pd.DataFrame([{
        "project_type": project_type,
        "team_size": team_size,
        "cloud_provider": cloud_provider,
        "budget": budget,
        "expected_users": expected_users,
        "deployment_target": deployment_target,
        "database_preference": database_preference,
        "experience": experience
    }])

    categorical_columns = [
        "project_type",
        "cloud_provider",
        "budget",
        "expected_users",
        "deployment_target",
        "database_preference",
        "experience"
    ]

    for col in categorical_columns:
        sample[col] = encoders[col].transform(sample[col])

    prediction = model.predict(sample)

    stack = encoders["stack"].inverse_transform(prediction)[0]

    return f"✅ Recommended Stack: {stack}"

# -------------------------------------
# Gradio UI
# -------------------------------------

demo = gr.Interface(
    fn=recommend_stack,
    inputs=[
        gr.Dropdown(
            choices=[
                "Web App",
                "API",
                "ML App",
                "Real-time App",
                "Mobile Backend"
            ],
            label="Project Type"
        ),

        gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Team Size"
        ),

        gr.Dropdown(
            choices=[
                "AWS",
                "Azure",
                "GCP",
                "On-Prem"
            ],
            label="Cloud Provider"
        ),

        gr.Dropdown(
            choices=[
                "Low",
                "Medium",
                "High",
                "Enterprise"
            ],
            label="Budget"
        ),

        gr.Dropdown(
            choices=[
                "Small",
                "Medium",
                "Large",
                "Massive"
            ],
            label="Expected Users"
        ),

        gr.Dropdown(
            choices=[
                "Docker",
                "Kubernetes",
                "VM",
                "Serverless"
            ],
            label="Deployment Target"
        ),

        gr.Dropdown(
            choices=[
                "SQLite",
                "PostgreSQL",
                "MongoDB",
                "Redis"
            ],
            label="Database Preference"
        ),

        gr.Dropdown(
            choices=[
                "Beginner",
                "Intermediate",
                "Expert"
            ],
            label="Team Experience"
        )
    ],

    outputs="text",

    title="Tech Stack Advisor",

    description=(
        "Get AI-powered technology stack recommendations "
        "based on your project requirements."
    ),
    allow_flagging="never"
)

# -------------------------------------
# Run app
# -------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
