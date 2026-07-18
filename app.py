import gradio as gr
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


def recommend_stack(
    project_type,
    team_size,
    cloud_provider,
    budget,
    expected_users,
    deployment_target,
    database_preference,
    experience,
):
    pt = encoders["project_type"].transform([project_type])[0]
    cp = encoders["cloud_provider"].transform([cloud_provider])[0]
    bd = encoders["budget"].transform([budget])[0]
    eu = encoders["expected_users"].transform([expected_users])[0]
    dt = encoders["deployment_target"].transform([deployment_target])[0]
    db = encoders["database_preference"].transform([database_preference])[0]
    ex = encoders["experience"].transform([experience])[0]

    input_data = np.array(
        [[
            pt,
            team_size,
            cp,
            bd,
            eu,
            dt,
            db,
            ex
        ]]
    )

    pred = model.predict(input_data)[0]

    stack = encoders["stack"].inverse_transform([pred])[0]

    return f"🔧 Recommended Tech Stack: {stack}"


demo = gr.Interface(
    fn=recommend_stack,
    inputs=[
        gr.Dropdown(
            ["Web App", "API", "ML App", "Real-time App", "Mobile Backend"],
            label="Project Type",
        ),
        gr.Slider(1, 20, step=1, label="Team Size"),
        gr.Dropdown(
            ["AWS", "Azure", "GCP", "On-Prem"],
            label="Cloud Provider",
        ),
        gr.Dropdown(
            ["Low", "Medium", "High", "Enterprise"],
            label="Budget",
        ),
        gr.Dropdown(
            ["Small", "Medium", "Large", "Massive"],
            label="Expected Users",
        ),
        gr.Dropdown(
            ["Docker", "Kubernetes", "VM", "Serverless"],
            label="Deployment Target",
        ),
        gr.Dropdown(
            ["SQLite", "PostgreSQL", "MongoDB", "Redis"],
            label="Database Preference",
        ),
        gr.Dropdown(
            ["Beginner", "Intermediate", "Expert"],
            label="Experience Level",
        ),
    ],
    outputs="text",
    title="Tech Stack Advisor",
    description="Get an AI-generated technology stack recommendation.",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)