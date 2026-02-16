import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración de página
st.set_page_config(page_title="Agente de Datos CSV", layout="wide")

# Inicialización de estado
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# Sidebar: Configuración
st.sidebar.header("Configuración")
provider = st.sidebar.radio("Proveedor de Modelo", ["Groq", "Google Gemini"])
api_key = st.sidebar.text_input("API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Cargar CSV", type=["csv"])

# Validación de entrada
if not uploaded_file or not api_key:
    st.warning("Por favor, ingresa tu API Key y sube un archivo CSV para continuar.")
    st.stop()

# Lógica de carga y persistencia del Agente
file_id = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.agent is None or st.session_state.last_file_id != file_id:
    try:
        # Cargar DataFrame
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        # Configurar LLM
        if provider == "Groq":
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                api_key=api_key, 
                temperature=0
            )
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key, 
                temperature=0
            )

        # Instrucción de Sistema (Hack de Visualización)
        PREFIX = """
        Eres un experto analista de datos. Tu objetivo es responder preguntas sobre el dataframe proporcionado.
        
        REGLA CRÍTICA PARA GRÁFICOS:
        Si se te pide generar un gráfico o visualización, SIEMPRE debes seguir estos pasos:
        1. Generar el código plotting usando matplotlib.
        2. Guardar la figura explícitamente como 'temp_plot.png' usando plt.savefig('temp_plot.png').
        3. NUNCA utilices plt.show() ya que no funciona en este entorno.
        4. Limpia la figura con plt.close() después de guardar.
        """

        # Crear Agente
        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            prefix=PREFIX,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
        
        st.session_state.last_file_id = file_id
        st.session_state.messages = [] # Reiniciar chat al cambiar archivo
        st.success("Agente inicializado correctamente.")
        
    except Exception as e:
        st.error(f"Error al inicializar el agente: {e}")
        st.stop()

# Interfaz de Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

# Lógica de Ejecución
if prompt := st.chat_input("Haz una pregunta sobre tus datos..."):
    # Renderizar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Ejecutar Agente
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analizando..."):
                response = st.session_state.agent.invoke({"input": prompt})
                output_text = response["output"]
                
                st.write(output_text)
                
                # Verificación y renderizado de gráficos
                image_path = "temp_plot.png"
                image_data = None
                
                if os.path.exists(image_path):
                    st.image(image_path)
                    # Leer para persistencia en historial si fuera necesario, o solo marcar ruta
                    # Aquí mostramos y luego eliminamos
                    image_data = image_path 
                    os.remove(image_path)
                
                # Guardar en historial
                msg_data = {"role": "assistant", "content": output_text}
                if image_data:
                    # Nota: Para persistencia real de imagen entre recargas de streamlit, 
                    # se debería guardar en bytes o cache, pero aquí seguimos el flujo lógico simple.
                    # Al eliminar el archivo, la imagen desaparecería del historial visual si se recarga la página
                    # a menos que se guarde en memoria.
                    pass 

                st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Error durante la ejecución: {e}")