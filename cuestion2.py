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
from langchain.document_loaders.merge import MergedDataLoader

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Create instance of OpenAI LLM

# user_api_key = st.sidebar.text_input(
#     label="#### Your OpenAI API key 👇",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")

prompt = st.sidebar.text_area("Prompt", value=f"""Sos un chatbot de una empresa,
     si la pregunta contiene información que no esta en la informacion del pdf responde
      'Lo siento no tengo informacion de ese producto, por favor contactate con ariel3.coman@hotmail.com/

Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags,i.e. ####.

Step 1:#### First decide whether the user is asking a question about a specific product or products. Product cateogry doesn't count.

Step 2:#### use the document info to check if you can answer the question with the information given, if the info does not answer the question


 check the following questions and answers: 
question: Buenas tardes, disculpe quisiera saber de cuantos kilos es el lavarropas que está en esta publicación, esperó su respuesta, gracias
Answer: Hola CHOSCORAUL69! Tiene una capacidad de 7kg. Contamos con stock de este producto, ofertá sin problemas! Ingresando tu código postal en el margen superior te va a indicar el costo y tiempo de entrega. Contamos con stock de este producto, ofertá sin problemas! Gracias por contactarte! Saludos! Oscar Barbieri S.A

question: Pero para lavar con agua caliente necesita estar conectado a agua caliente? O el lavarropas calienta el agua?
Answer: ¡Hola! te escribe Paula, gracias por comunicarte. Disculpe el fabricante oficial no lo indica , lo estamos verficando. Seguimos en línea para responder tus consultas. Saludos, Equipo DMake

question:Hola, la altura es de 85cm? sabes cual es el minimo regulando las patas? porque necesito que tenga menos de 84cm
Answer:¡Hola! En este momento lamentamos no estar disponibles para atender tu consulta, podes comunicarte con nosotros nuevamente desde las 14hs. ¡Gracias por escribirnos

question: Hola una consulta para lavar con agua caliente es necesario conectarlo con agua caliente o el lavarropa calienta el agua fría?·
Answer: El lavarropas calienta el agua

Step 4:####: Any questions related to the height, width and depth of the product or the space the products ocupies only respond the dimensions in cm and nothing else

Step 5:####: If the user made any assumptions, figure out whether the assumption is true based on your product information.

Step 6:####: First, politely correct the customer's incorrect assumptions if applicable. 
Answer the customer in a friendly tone.

Use the following format:
Step 1:#### <step 1 reasoning>
Step 2:#### <step 2 reasoning>
Step 3:#### <step 3 reasoning>
Step 4:#### <step 4 reasoning>
Step 5:#### <step 5 reasoning>
Step 6:#### <step 6 reasoning>
Response to user:#### <response to customer>

Make sure to include #### to separate every step.


      """)
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")


uploaded_file_csv = st.sidebar.file_uploader("upload", type="csv")

delimiter = "####"


if uploaded_file and uploaded_file_csv:
    # use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(file_path=tmp_file_path)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file_csv:
        tmp_file_csv.write(uploaded_file_csv.getvalue())
        tmp_file_path_csv = tmp_file_csv.name

    loader_csv = CSVLoader(file_path=tmp_file_path_csv, encoding="utf-8", csv_args={'delimiter': ','})
    # loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
    # 'delimiter': ','})

    loader_all = MergedDataLoader(loaders=[loader, loader_csv])
    data = loader_all.load()
    # embeddings = HuggingFaceEmbeddings()

    hfemb = HuggingFaceEmbeddings()
    embeddings = LangchainEmbedding(hfemb)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings)
    # Create vectorstore info object - metadata repo?
    #     repo_id = "google/flan-t5-xl"
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
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
        st.session_state['generated'] = [
            "Hola mi nombre es Ariel de parte de impala soluciones ¿en que puedo ayudarte? 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

        # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()
    # prompt = """
    #
    #   """
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(prompt + user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")