import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


load_dotenv()

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))

tools = load_tools(['ddg-search','wikipedia'], llm = llm)

# print(tools[1].name, tools[1].description)

prompt = hub.pull('hwchase17/react')

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)

query = """"
Vou viajar para Tailandia em Julho de 2025. Quero que faça um roteiro de viagem para mim. com eventos que irão ocorrer na data da viagem e com o preço de passagem mais em conta a partir do Rio de Janeiro.
"""

agent_executor.invoke({"input": query})