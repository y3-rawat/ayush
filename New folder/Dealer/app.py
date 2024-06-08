
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from langchain_community.llms import Ollama
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for session management

key = "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"
DB_FAISS_PATH = 'dealer/store/db_faiss'

# Load the model
def load_llm():
    # llm = ChatGroq(model="llama3-8b-8192", api_key=key)
    llm = Ollama(model="llama3")
    return llm

loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def prompt_template(query, unique_id):
    
    
    df = pd.read_json("dealer.json")
    d = df.where(df.dealer_id == unique_id).dropna().iloc[0]
    
    template = f"""You are Eurotech Xchange Bot, a specialized assistant with detailed information about dealer activities and data. You have access to the following data about the dealer:
    As the Eurotech Xchange Bot, you can answer any questions related to this collector's car activities, preferences, bid amounts, or any other related information.
    , information about the data
    if you don't know any thing then give the email and said that you can connect with team our team will help you :
    email : info@eurotechxchange.com
    
    if user is just doing casual chat then entertain him by answer to his question or chat 

    You have access to the following data about a specific collector:

    Booked Car: {d["car_added"]}
    how many peoples are visited :{d["peoples_visited"]}
    Logs:
    The visiters have visited the following cars with count: {d["visited_logs"]}
    
    this is a data about bids on cars and with which amount and car user has placed bid 
    Bid in car and Amount:
    {d["bid_placed_on_cars"]}

    total bids on car is : {d["Total bid placed on cars"]}
    As the Eurotech Xchange Bot, you can answer any questions related to the dealer's sales  
     
    You will provide the relevant information based on the data available.
    
    if user is just doing casual chat then just do casual chatting about the cars
    ---------------
        
    User_Chat: {query}
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
        session['generated'] = ["Hello! Ask me anything about 🤗"]
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
