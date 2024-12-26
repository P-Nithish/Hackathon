# Chatbot Application

## Problem Statement:
Design and implement a chatbot system capable of ingesting and interpreting
uploaded documents (e.g., PDFs) to provide accurate, fact-based responses
quickly and reliably. The chatbot should utilize LLM APIs and other retrieval
techniques.


## Introduction
This project is a Flask-based chatbot application that interfaces with a MongoDB database and uses the Groq API to handle natural language processing tasks. It's designed to extract text from uploaded files, generate responses to user queries based on the extracted text, and maintain a chat history for each user.

## Modules / Libraries Used
- Flask: A lightweight WSGI web application framework.
- MongoDB: NoSQL database used for storing user data and chat history.
- PyPDF2: A Pure-Python library built as a PDF toolkit.
- python-docx: Reads, queries and modifies Microsoft Word docx files.
- Groq: AI and ML API used for processing natural language queries.
- Flask-Bcrypt: Provides bcrypt hashing utilities for your application.

## Features Implemented
- User authentication (signup, signin, signout).
- File upload and text extraction from `.txt`, `.pdf`, and `.docx` files.
- Support bulk upload and processing
- Text summarization to adhere to token limits of the NLP model.
- Interaction with Groq API to generate chat responses based on the uploaded content.
- Persistent chat history for each user session stored in MongoDB.

## How to Run

### Setting Up the Environment

1. **Clone the Repository**
2. **Create and Activate Virtual Environment**
    ```
    python -m venv flask_env  
    flask_env\Scripts\activate
    ```
3. **Install Requirements**
   ```
       pip install -r requirements.txt
    ```
## How to Use
1. Navigate to the sign-up page to create a new user account.
2. Log in using your credentials.
3. Upload text, PDF, or Word documents.
4. Enter questions to get responses based on the content of the uploaded files.
5. Review your chat history anytime during the session.

## Additional Information
- Ensure MongoDB is running on your system to store user data and chat history.
- The application is configured to use default settings for MongoDB (`localhost:27017`). Adjust these in your configuration if necessary.

## Output
---
<img src="https://github.com/P-Nithish/Hackathon/blob/main/img1.png">
<img src="https://github.com/P-Nithish/Hackathon/blob/main/img2.png">
<img src="https://github.com/P-Nithish/Hackathon/blob/main/image.png">

