import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

st.set_page_config(page_title="Data Pro Agent", layout="wide")

st.title("📊 Agente de Análisis de Datos Pro")

with st.expander("Guía de Uso", expanded=True):
    st.info("Este agente analiza tus CSV, genera gráficos y busca en internet si necesitas contexto adicional. Sube un archivo, introduce tu API Key y empieza a preguntar.")

with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file and api_key:
    try:
        df = pd.read_csv(uploaded_file)
        
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=api_key, 
            model_name="llama-3.3-70b-versatile"
        )
        
        search_tool = DuckDuckGoSearchRun()
        
        # Mejoramos el prompt para el sistema ReAct
        custom_prefix = """You are a professional data analyst. 
        You have access to a pandas dataframe (df) and a search tool.
        If the user asks for a chart, you MUST save it as 'temp_plot.png' using matplotlib and confirm it in your final answer.
        For external context, use the search tool.
        Always use the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of [python_repl_ast, duckduckgo_search]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        """

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="zero-shot-react-description", # Cambio crítico para estabilidad
            allow_dangerous_code=True,
            extra_tools=[search_tool],
            prefix=custom_prefix,
            handle_parsing_errors=True # Crucial para manejar salidas mal formateadas
        )

        if prompt := st.chat_input("¿Qué quieres saber de tus datos?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Pensando y analizando..."):
                        # Capturamos la respuesta
                        full_response = agent.invoke({"input": prompt})
                        answer = full_response["output"]
                        
                        st.markdown(answer)
                        
                        # Detección de uso de internet en el razonamiento
                        if "duckduckgo_search" in str(full_response).lower():
                            st.caption("🔍 Información contrastada en internet")
                        
                        if os.path.exists("temp_plot.png"):
                            st.image("temp_plot.png")
                            os.remove("temp_plot.png")
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Error en la ejecución: {str(e)}")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.warning("Introduce la API Key y sube un CSV para comenzar.")