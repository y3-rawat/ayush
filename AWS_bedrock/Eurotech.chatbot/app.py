# Importing necessary modules

from flask import Flask, render_template, jsonify, redirect, url_for, request
import small_work as main1
import string 
import gpt
from difflib import SequenceMatcher
from flask_cors import CORS
import json
# Creating a Flask app instance
app = Flask(__name__)
first_time = True
CORS(app, resources={r"/*": {"origins": '*'}})
def get_response(chatbot, user_input):
    print("User Input:", user_input)  # Add this line for debugging

    
    user_input = user_input.lower().translate(str.maketrans('', '', string.punctuation))
    max_similarity = 0
    best_match = None
    for question in chatbot:
        similarity = SequenceMatcher(None, user_input, question).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = question
    if max_similarity >= 0.7:
        return chatbot[best_match]
    else:
        ans = gpt.userQuestion(user_input)
        return ans["response"]
@app.route('/redirect_to_second')
def redirect_to_second():
    return redirect(url_for('second_page'))

# Route for the second page


# Defining the route for the home page
@app.route("/")
def index():
    return render_template("index.html")


# Defining the route for getting bot response
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    chatbot = main1.create_chatbot()
    if userText in chatbot:
        response = chatbot[userText]
        return response
    else:
        response = gpt.userQuestion(userText)
        print(response,"respoonce")
        ans = gpt.main_module_for_genrating_answer(userText,response)
        return ans.content
        # try:
        #     response = get_response(chatbot, userText)
        #     return response
        # except Exception as e:
        #     try:
        #         response = get_response(chatbot, userText)
        #         return response
        #     except:
        #         try:
        #             response = get_response(chatbot, userText)
        #             return response
        # except:
        #     return f"An error occurred: Trying again"

if __name__ == "__main__":
    app.run(debug=True)
