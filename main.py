from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever


# COMPONENTE RAG: GENERATION (PARTE 1) - CONFIGURACIÓN DEL MODELO LLM
# Configuramos el modelo de lenguaje que generará las respuestas
# llama3.2 es el modelo generativo que creará respuestas basadas en el contexto recuperado
model = OllamaLLM(model="llama3.2", temperature=0.7)

template = """
You are an expert in answering questions about Medellin, Colombia.

The data below is structured as:
[Name] [Area_km2] [Population] [Favorite_Food] [Notable_Landmark] [Average_Income_USD]

Data:
{reviews}

Question: {question}

Please provide a detailed answer based on the data provided.
Write at least a paragraph explaining the answer and providing additional relevant information.
Make sure to clearly identify which location you're referring to when answering the question.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


while True:
    print("\n\n----------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    # COMPONENTE RAG: RETRIEVAL (PARTE 4) - EJECUCIÓN DE LA RECUPERACIÓN
    # Aquí se invoca al retriever con la pregunta del usuario
    # El retriever busca en la base de datos vectorial los documentos más similares
    docs = retriever.invoke(question)

    print("---Documentos Recuperados---")
    if docs:
        for i, doc in enumerate(docs, start=1):
            print(f"Documento {i}:")
            print(doc.page_content)
            print(f"Metadatos: {doc.metadata}")
            print("---")
    else:
        print("No se recuperaron documentos.")
    print("---------------------------------")

    # COMPONENTE RAG: AUGMENTATION - INSERCIÓN DE CONTEXTO RECUPERADO
    # Aquí los documentos recuperados se formatean para ser agregados al prompt
    # Este es el paso clave donde "aumentamos" el prompt con información relevante
    reviews_text = "\n\n".join([doc.page_content for doc in docs])

    # COMPONENTE RAG: GENERATION (PARTE 2) - GENERACIÓN DE RESPUESTA
    # Aquí se invoca al modelo LLM pasándole:
    # 1. La pregunta original del usuario
    # 2. Los documentos relevantes recuperados (contexto aumentado)
    # El modelo genera una respuesta basada en este contexto enriquecido
    result = chain.invoke({
        "reviews": reviews_text,
        "question": question
    })
    print(result)
