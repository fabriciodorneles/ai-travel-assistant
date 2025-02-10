import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))

tools = load_tools(['ddg-search','wikipedia'], llm = llm)

# print(tools[1].name, tools[1].description)

agent = initialize_agent(
    tools,
    llm,
    agent='zero-shot-react-description',
    verbose = True
)

print(agent.agent.llm_chain.prompt.template)

query = """"
Vou viajar para Tailandia em Julho de 2025. Quero que faça um roteiro de viagem para mim. com eventos que irão ocorrer na data da viagem e com o preço de passagem mais em conta a partir do Rio de Janeiro.
"""

agent.run(query)