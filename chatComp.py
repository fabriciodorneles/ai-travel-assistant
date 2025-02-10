from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Initialize console and client
console = Console()
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user","content": "Vou viajar para Tailandia em Junho de 2025. Quero que faça um roteiro de viagem para mim."}
    ]
)

response = completion.choices[0].message.content
markdown_text = response.replace('\n', '\n\n')  # Double line breaks for Markdown
console.print(Markdown(markdown_text))