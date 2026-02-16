import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Agente de Datos CSV", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

st.title("📊 AI Data Insights Agent")
st.markdown("""
Esta interfaz permite interactuar con archivos CSV mediante lenguaje natural utilizando agentes de IA avanzados.
* **Capacidades:** Análisis descriptivo, limpieza lógica de datos y generación de visualizaciones.
* **Motores:** Llama 3.3 70B (Groq) o Gemini 1.5 Flash (Google).
* **Instrucción:** Sube un archivo en el menú lateral y formula preguntas complejas o solicita gráficos.
""")
st.divider()

st.sidebar.header("Configuración")
provider = st.sidebar.radio("Proveedor de Modelo", ["Groq", "Google Gemini"])
api_key = st.sidebar.text_input("API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Cargar CSV", type=["csv"])

if not uploaded_file or not api_key:
    st.warning("Configuración pendiente: Ingresa la API Key y carga un CSV en la barra lateral.")
    st.stop()

file_id = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.agent is None or st.session_state.last_file_id != file_id:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        
        if provider == "Groq":
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)

        PREFIX = """
        Eres un experto analista de datos. Tu objetivo es responder preguntas sobre el dataframe proporcionado.
        
        REGLA CRÍTICA PARA GRÁFICOS:
        Si se te pide generar un gráfico o visualización, SIEMPRE debes seguir estos pasos:
        1. Generar el código plotting usando matplotlib.
        2. Guardar la figura explícitamente como 'temp_plot.png' usando plt.savefig('temp_plot.png').
        3. NUNCA utilices plt.show() ya que no funciona en este entorno.
        4. Limpia la figura con plt.close() después de guardar.
        """

        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            prefix=PREFIX,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
        
        st.session_state.last_file_id = file_id
        st.session_state.messages = []
        st.success("Agente listo para análisis.")
        
    except Exception as e:
        st.error(f"Error en inicialización: {e}")
        st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

if prompt := st.chat_input("Consulta técnica sobre el dataset..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Procesando consulta..."):
                response = st.session_state.agent.invoke({"input": prompt})
                output_text = response["output"]
                st.write(output_text)
                
                image_path = "temp_plot.png"
                msg_data = {"role": "assistant", "content": output_text}
                
                if os.path.exists(image_path):
                    st.image(image_path)
                    # En entornos de producción, aquí deberías codificar a base64 para persistir en session_state
                    os.remove(image_path)
                
                st.session_state.messages.append(msg_data)

        except Exception as e:
            st.error(f"Error de ejecución: {e}")