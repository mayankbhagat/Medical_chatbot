pip install ibm-watsonx-ai

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#  YOUR CREDENTIALS GO HERE
api_key = "3zSo_0kyFARZrGQjT1ZqVV41ZSFqaiIfzYMRrNW0XTsW"
project_id = "333f4af9-4dfc-473a-8663-964eb517c51a"
watsonx_url = "https://us-south.ml.cloud.ibm.com"

#  Authenticator
authenticator = IAMAuthenticator(api_key)

#  Model Inference Client
model = ModelInference(
    model_id="ibm/granite-13b-chat-v2",
    project_id=project_id,
    url=watsonx_url,
    authenticator=authenticator
)

#  Prompt
def build_prompt(symptom):
    return f"""
You are a medical assistant. Based on this symptom, recommend the correct department.
Symptom-to-Department Mapping:
- chest pain: Cardiology
- palpitations: Cardiology
- skin rash: Dermatology
- itching: Dermatology
- headache: Neurology
- dizziness: Neurology
- shortness of breath: Pulmonology
- persistent cough: Pulmonology
- joint pain: Orthopedics
- back pain: Orthopedics

Symptom: {symptom}
Return only the department name.
"""

# Generate
def get_department(symptom):
    prompt = build_prompt(symptom)
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text'].strip()

#  Run
if __name__ == "__main__":
    user_input = input("Enter your symptom: ")
    department = get_department(user_input)
    print("Recommended department:", department)
