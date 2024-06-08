from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
# from langchain_community.llms import Ollama
import functools

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for session management

key = "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"
DB_FAISS_PATH = 'collector/store/db_faiss'

def load_llm():
    llm = ChatGroq(model="llama3-8b-8192", api_key=key)
    # llm = Ollama(model="llama3")

    return llm

def load_data():
    loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)

    return db

db = load_data()
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

@functools.lru_cache(maxsize=128)
def prompt_template(query, unique_id):
    
    df = pd.read_json("collector_data.json")
    d = df.where(df.user_id == unique_id).dropna().iloc[0]
    
    template = f"""You are Eurotech Xchange Bot, a specialized assistant with detailed information about car collectors and their activities.
    We also take Digital Currencys like Bitcoin etc.
    if user ask any thing which is complex then just give the email and said that you can connect with team our team will help you
    email : info@eurotechxchange.com
    You have access to the following data about a specific collector:

    Booked Car: {d["Booked_car"]}
    Price of Car:{d["Price_of_car"]}
    Logs:
    The collector has visited the following cars: {d["logs"][0]["visit_cars"]}
    used has placed bid on these cars and with these amounts
    Bid in Amount:
    {d["bid_in_amount"]}

    As the Eurotech Xchange Bot, you can answer any questions related to this collector's car activities, preferences, bid amounts, or any other related information.
    Feel free to ask questions about the collector's booked car, visited cars, bid amounts, or any other related inquiries.We also take Digital Currencys like bitcoin etc.

     Answer user with less then 50 words
     -------
     Question: {query}
     """
    return template

def conversational_chat(query, unique_id):
    prompt = prompt_template(query, unique_id)
    result = chain({"question": prompt, "chat_history": session.get('history', [])})
    session['history'].append((query, result["answer"]))
    return result["answer"]

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    if 'generated' not in session:
        session['generated'] = ["Hello! Ask me anything 🤗"]
    if 'past' not in session:
        session['past'] = ["Hey! 👋"]
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['query']
    unique_id = int(request.form['unique_id'])
    if user_input:
        output = conversational_chat(user_input, unique_id)
        session['past'].append(user_input)
        session['generated'].append(output)
    return jsonify({'past': session['past'], 'generated': session['generated']})

@app.route("/get", methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    unique_id = int(request.args.get('unique_id'))
    if user_text:
        output = conversational_chat(user_text, unique_id)
        return output
    return "Please provide a query parameter ?msg=Your_MSG and unique_id parameter ?unique_id=Your_ID"

if __name__ == "__main__":
    app.run(debug=True)
