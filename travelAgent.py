import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

import bs4

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence


load_dotenv()

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))

query = """"
Vou viajar para Tailandia em Julho de 2025. Quero que faça um roteiro de viagem para mim. com eventos que irão ocorrer na data da viagem e com o preço de passagem mais em conta a partir do Rio de Janeiro.
"""

def researchAgent(query, llm):
    tools = load_tools(['ddg-search','wikipedia'], llm = llm)
    prompt = hub.pull('hwchase17/react')
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        prompt=prompt,
        handle_parsing_errors=True,  # Add this
        max_iterations=3,            # Prevent infinite loops
        verbose=True
    )
    
    try:
        webContext = agent_executor.invoke({"input": query})
        return webContext['output']
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def loadData():
    try:
        # Fix URL format
        url = "https://www.dicasdeviagem.com/inglaterra/"
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("title entry-title", "postcontentwrap")
                )
            ),
        )
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        # Fix: Use split_documents instead of split
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            print("No documents to process")
            return None
            
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()
        )

        return vectorstore.as_retriever()
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def getRelevantDocs(query: str):
    retriever = loadData()
    if not retriever:
        print("Failed to initialize retriever")
        return None
    try:
        response = retriever.get_relevant_documents(query)
        print(retriever.get_relevant_documents(query))
        return response
    except Exception as e:
        print(f"Error getting relevant docs: {str(e)}")
        return None

def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro de viagem.
    Contexto: {webContext}
    Documento Relevantes: {relevant_documents}
    Usuário: {query}
    Assistente:
    """

    prompt = PromptTemplate(
        input_variables= ["webContext", "relevant_documents", "query"],
        template = prompt_template
    )

    sequence = RunnableSequence(prompt | llm)

    response = sequence.invoke({
        "webContext": webContext, 
        "relevant_documents": relevant_documents, 
        "query": query
    })

    return response

def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response

print(getResponse(query, llm).content)
    