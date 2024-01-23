# app.py (combined)
from flask import Flask, render_template, request
from flask_cors import CORS
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

urls = [
    "https://marg-darshan.com/",  # margdarshan website
    "https://jeemain.nta.nic.in/about-jeemain-2023/",  # Govt NTA
    "https://byjus.com/jee/jee-main/",
    "https://engineering.careers360.com/articles/jee-main-2024"
]

file_path = "faiss_store_openai.pkl"

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=500)

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ','],
    chunk_size=1000
)
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectorstore_openai = FAISS.from_documents(docs, embeddings)

with open(file_path, "wb") as f:
    pickle.dump(vectorstore_openai, f)


@app.route('/')
def index():
    return render_template('./home.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.get_json()
    user_query = user_input["user_input"]

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": user_query}, return_only_outputs=True)
            response = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", "")
            }
            return response
    else:
        return {"error": "FAISS index not found"}


if __name__ == '__main__':
    app.run(debug=True)
