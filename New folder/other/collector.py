import pandas as pd
import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
key = "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"

DB_FAISS_PATH = 'vectorstore/collector/db_faiss'

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = ChatGroq(model="llama3-8b-8192", api_key=key)
    return llm


st.markdown("<h3 style='text-align: center; color: white;'></h3>", unsafe_allow_html=True)

loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def prompt_template(query):
    unique_id = 1
    df = pd.read_json("data.json")
    d = df.where(df.user_id == unique_id).dropna().iloc[0]
    template = f"""You are Eurotech Xchange Bot, a specialized assistant with detailed information about car collectors and their activities. 
You have a Access of database which is on frontend
    You have access to the following data about a specific collector:

    Booked Car: {d["Booked_car"]}
    Price of Car:{d["Price_of_car"]}
    Logs:
    The collector has visited the following cars: {d["logs"][0]["visit_cars"]}
    used has placed bid on these cars and with these amounts
    Bid in Amount:
    {d["bid_in_amount"]}
    As the Eurotech Xchange Bot, you can answer any questions related to this collector's car activities, preferences, bid amounts, and other related information.

    Feel free to ask questions about the collector's booked car, visited cars, bid amounts, or any other related inquiries.


    Question: {query}
    """
    return template

def conversational_chat(query):
    prompt = prompt_template(query)
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Container for the chat history
response_container = st.container()
# Container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
