# Local-Agent RAG con ChromaDB, LangChain y Ollama

Este repositorio muestra cómo montar un pipeline de **Retrieval-Augmented Generation (RAG)** usando:

- **Python 3.12.x**  
- **uv** como gestor de entornos y dependencias  
- **ChromaDB** para vector search local  
- **LangChain** para orquestar embeddings y chains  
- **Ollama** como LLM y proveedor de embeddings

## 📝 Descripción del Proyecto

Este proyecto implementa un asistente basado en RAG que responde preguntas sobre localidades en Medellín, Colombia. La aplicación:

1. **Carga datos locales** sobre comunas y municipios (nombre, área, población, comida favorita, puntos de interés, ingresos)
2. **Utiliza embeddings** para convertir texto en vectores mediante el modelo `mxbai-embed-large`
3. **Almacena los datos en ChromaDB** como base de datos vectorial local
4. **Recupera información relevante** cuando el usuario hace preguntas específicas sobre las localidades
5. **Genera respuestas contextualizadas** usando el LLM `llama3.2`

Todo funciona de manera local y sin conexión a internet gracias a Ollama.

> **Nota**: Este repositorio es una adaptación del tutorial presentado en [este video de Tech With Tim](https://www.youtube.com/watch?v=E4l91XKQSgw&t=502s&ab_channel=TechWithTim).

---

## 📋 Requisitos

- **Python ≥ 3.12, < 4.0**  
- **curl** (para instalar uv)  
- **Git** (para clonar el repositorio)

---

## ⚙️ Instalación de `uv`

Sigue la guía oficial de uv: https://docs.astral.sh/uv/#getting-started  

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Una vez que `uv` esté instalado y disponible en tu terminal, puedes navegar al directorio raíz del curso y ejecutar:

```bash
# Primero crea el entorno virtual con Python 3.12
uv venv --python 3.12

# Luego instala las dependencias desde uv.lock
uv sync
```

Esto:

* Crea un entorno virtual .venv/ con Python 3.12.x
* Instala las mismas versiones bloqueadas en uv.lock

Para activar el ambiente virtual:

```bash
# En macOS o Linux
source .venv/bin/activate

# En Windows
# .venv\Scripts\activate
```

Una vez activado, verás el nombre del entorno (.venv) al inicio de tu línea de comandos.

Recuerda siempre usar `uv` antes de `pip install`.

## 🧩 Componentes Principales

### 1. Vector Search con ChromaDB

La búsqueda vectorial se implementa localmente usando ChromaDB, permitiendo:
- Convertir texto a vectores (embeddings)
- Almacenar eficientemente estos vectores
- Buscar información por similitud semántica

### 2. Pipeline RAG

El sistema utiliza un pipeline RAG (Retrieval-Augmented Generation) compuesto por:

- **Retrieval**: Recupera documentos relevantes usando similaridad de embeddings
  - Utiliza `mxbai-embed-large` para generar representaciones vectoriales
  - Almacena y consulta vectores en ChromaDB
  - Configura un retriever para devolver los 5 documentos más similares

- **Augmentation**: Enriquece la consulta del usuario con el contexto recuperado
  - Formatea los documentos recuperados para insertarlos en el prompt

- **Generation**: Genera respuestas usando el LLM
  - Utiliza el modelo `llama3.2` para crear respuestas naturales
  - Incorpora el contexto recuperado para responder con precisión

## 🚀 Uso de la Aplicación

Una vez configurado el entorno, ejecuta:

```bash
python main.py
```

La aplicación mostrará un prompt donde podrás preguntar sobre las diferentes localidades de Medellín, por ejemplo:
- "¿Cuál es la comida favorita en Robledo?"
- "¿Qué lugares turísticos hay en El Poblado?"
- "¿Cuál es el ingreso promedio en Envigado?"

La aplicación recuperará los documentos relevantes y generará una respuesta basada en los datos disponibles.