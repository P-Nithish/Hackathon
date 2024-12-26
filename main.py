from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
import os
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a secure key in production
bcrypt = Bcrypt(app)

# Configure Groq API Client
client = Groq(api_key="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")

# Connect to MongoDB
client_db = MongoClient("mongodb://localhost:27017/")
db = client_db.chatbot_app
users_collection = db.users

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store extracted text
parsed_data = ""

# Function to extract text from files
def extract_text_from_file(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    elif filepath.endswith(".pdf"):
        pdf_reader = PdfReader(filepath)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        raise ValueError("Unsupported file type")

# Home Route (Sign-in Page)
@app.route("/")
def index():
    if 'username' in session:
        return redirect(url_for('upload_page'))
    return render_template("signin.html")

# Sign-up Page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if username already exists
        if users_collection.find_one({"username": username}):
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))

        # Hash the password and store the user
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        users_collection.insert_one({"username": username, "password": hashed_password})
        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('index'))

    return render_template("signup.html")

# Sign-in Route
@app.route("/signin", methods=["GET","POST"])
def signin():
    username = request.form.get("username")
    password = request.form.get("password")

    user = users_collection.find_one({"username": username})
    if user and bcrypt.check_password_hash(user["password"], password):
        session['username'] = username  # Set the session
        return redirect(url_for('upload_page'))
    flash('Invalid credentials', 'error')
    return redirect(url_for('index'))

# Sign-out Route
@app.route("/signout")
def signout():
    session.pop('username', None)  # Clear the session
    global parsed_data
    parsed_data = ""
    return redirect(url_for('index'))

# Fetch Chat History Route
@app.route("/chat_history", methods=["GET"])
def chat_history():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = users_collection.find_one({"username": session['username']})
    if not user or "chats" not in user:
        return jsonify({"chats": []})  # Return an empty list if no chat history exists

    return jsonify({"chats": user["chats"]}), 200


# File Upload Page
@app.route("/upload_page")
def upload_page():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Retrieve chat history for the logged-in user
    user = users_collection.find_one({"username": session['username']})
    chat_history = user.get("chats", []) if user else []
    
    return render_template("upload.html", chat_history=chat_history)


# Upload Route
@app.route("/upload", methods=["POST"])
def upload():
    global parsed_data  # This holds all text from all uploaded files
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    files = request.files.getlist("file")  # Get list of files from the form
    if not files:
        return jsonify({"error": "No file uploaded"}), 400

    for file in files:
        if file:  # Make sure a file is present
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                file_text = extract_text_from_file(filepath)
                parsed_data += file_text + "\n"  # Append new text to the existing content
            except Exception as e:
                return jsonify({"error": f"Failed to extract content from file {filename}: {str(e)}"}), 500

    return jsonify({"message": "Files uploaded and parsed successfully!"}), 200

def reduce_text_size(text, max_tokens=3000):
    sentences = text.split('.')
    reduced_text = ""
    token_count = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if token_count + sentence_tokens > max_tokens:
            break
        reduced_text += sentence + '.'
        token_count += sentence_tokens
    
    return reduced_text

# Chatbot Route
@app.route("/chat", methods=["POST"])
def chat():
    global parsed_data
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"})

    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"})

    if not parsed_data:
        return jsonify({"error": "No file content available. Please upload a file first."})
    
    parsed_data = reduce_text_size(parsed_data)

    # Use Groq API with Llama model to generate a response based on file content and question
    prompt = f"""
    You are an intelligent assistant that answers questions accurately and concisely based only on the given context. 
    Do not include any information that is not explicitly provided in the context below.

    Context: 
    {parsed_data}

    Question: {question}

    Only respond using the context above. If the context does not contain the answer, reply with 
    "I'm sorry, the provided context does not contain sufficient information to answer this question."
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
        db.users.update_one(
        {"username": session['username']},
        {"$push": {"chats": {"question": question, "answer": answer}}},
        upsert=True
    )
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"Failed to generate a response: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)