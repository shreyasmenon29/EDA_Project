from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

print("=" * 80)
print("AVAILABLE GEMINI MODELS")
print("=" * 80)

for model in client.models.list():
    if model.supported_actions and "generateContent" in model.supported_actions:
        print(model.name)