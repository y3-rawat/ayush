
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq


app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for session management


DB_FAISS_PATH = 'dealer/store/db_faiss'


def load_llm():
    
    llm = ChatGroq(model="llama3-8b-8192", api_key="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")
    return llm

loader = CSVLoader(file_path="data_car.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def prompt_template(query):
    df = pd.read_json("Admin_database.json")
    template = f"""
    --- Eurotech Xchange Bot: Admin Support ---
    Role: Provide comprehensive assistance and data retrieval for dealership operations as connected to the admin panel.
    Capabilities: Access and analyze detailed dealership data, including transactions, inventory, user interactions, and performance metrics.

    Data Accessibility:
    - Total Cars in Inventory: {df["total_cars_in_inventory"]} cars
    - Total User Visits: {df["total_users_visited"]} visits
    - IDs of Repeat Visitors: {df["Repetative_Users_id"]}
    - Most Active User: {df["Most_car_viewed_user_id"]}
    - Number of Dealers: {df["Dealers_on_webiste"]}
    - Transaction History Overview: {df["Transections_history"]}
    - Total Cars Sold: {df["Car_Sold_total"]} cars
    - Last Month's Car Sales: {df["Car_sold_last_month"]} cars
    - Top 5 Most Viewed Cars: {df["top_5_cars_viewd"]}
    - Common Website Issues: {df["common_issues_on_webiste"]}
    - Cars with Most Bids: {df["most_bids_of_cars"]}

    Guidelines:
    - Maintain a professional and respectful tone in all interactions.
    - Provide precise and actionable data insights.
    - Adjust the language based on the user's inquiry and the seriousness of the discussion.
    - Ensure responses are concise, ideally fewer than 50 words, unless detailed explanation is necessary.

    Security Note:
    - Handle all data with strict confidentiality, especially when dealing with sensitive information.
    
    Current Inquiry: {query}
    """
    return template


def conversational_chat(query):
    prompt = prompt_template(query)
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
    
    if user_input:
        output = conversational_chat(user_input)
        session['past'].append(user_input)
        session['generated'].append(output)
    return jsonify({'past': session['past'], 'generated': session['generated']})

@app.route("/get", methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    
    if user_text:
        output = conversational_chat(user_text)
        return output
    return "Please provide a query parameter ?msg=Your_MSG and unique_id parameter ?unique_id=Your_ID"

if __name__ == "__main__":
    app.run(debug=True)

