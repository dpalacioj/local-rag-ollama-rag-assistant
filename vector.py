from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Cargamos los datos de las reviews
df = pd.read_csv("amva-reviews.csv")

# COMPONENTE RAG: RETRIEVAL (PARTE 1) - EMBEDDINGS
# Configuramos el modelo de embeddings que convertirá el texto en vectores
# mxbai-embed-large es el modelo que genera representaciones vectoriales de alta calidad
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Create a formatted page content with proper string conversion
        content = (f"{str(row['Name'])} {str(row['Area_km2'])} {str(row['Population'])} "
                  f"{str(row['Favorite_Food'])} {str(row['Notable_Landmark'])} "
                  f"{str(row['Average_Income_USD'])}")
        
        document = Document(
            page_content=content,
            metadata={"Location-Type": row["Location_Type"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create the Chroma vector store

# COMPONENTE RAG: RETRIEVAL (PARTE 2) - BASE DE DATOS VECTORIAL
# Creamos/conectamos a la base de datos vectorial Chroma que almacenará los documentos y sus embeddings
vector_store = Chroma(
    collection_name="AMVA-Reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Si es la primera vez, añadimos los documentos a la base de datos
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()

# COMPONENTE RAG: RETRIEVAL (PARTE 3) - CONFIGURACIÓN DEL RETRIEVER
# Configuramos el componente que se encarga de recuperar documentos similares
# k=5 significa que recuperará los 5 documentos más similares a la consulta
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
