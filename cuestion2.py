import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from llama_index import LangchainEmbedding
from streamlit_chat import message
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.document_loaders import CSVLoader, PyPDFLoader
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
# Bring in streamlit for UI/app interface
import tempfile
from langchain.indexes import VectorstoreIndexCreator
# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.vectorstores import FAISS


# Create instance of OpenAI LLM

# user_api_key = st.sidebar.text_input(
#     label="#### Your OpenAI API key 👇",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)
    # loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
    # 'delimiter': ','})
    data = loader.load()
    # embeddings = HuggingFaceEmbeddings()

    hfemb = HuggingFaceEmbeddings()
    embeddings = LangchainEmbedding(hfemb)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings)
# Create vectorstore info object - metadata repo?
#     repo_id = "google/flan-t5-xl"
    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever())
    # chain = ConversationalRetrievalChain.from_llm(
    # llm=HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64}),
    # retriever=vectorstore.as_retriever())


    def conversational_chat(query):
        result = chain({"question": query,
                        "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))

        return result["answer"]


    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hola mi nombre es Ariel de parte de impala soluciones ¿en que puedo ayudarte? 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

        # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()
    prompt = """Sos un chatbot de una empresa,
     si la pregunta contiene información que no esta en la informacion del pdf responde
      'Lo siento no tengo informacion de ese producto, por favor contactate con ariel.coman@hotmail.com
      
      """
    with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(prompt+user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")