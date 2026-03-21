from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")
print("KEY EXISTS:", bool(api_key))
print("KEY PREFIX:", api_key[:7] if api_key else None)

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5",
    input="Spune doar: merge"
)

print(response.output_text)