import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")

with open("rag_dataset_combined.jsonl") as f:
    chunks = [json.loads(line) for line in f]

def retrieve(query, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        overlap = query_words & set(chunk["text"].lower().split())
        scored.append((len(overlap), chunk["text"]))
    scored.sort(reverse=True)
    return "\n\n".join(text for _, text in scored[:top_k])

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message")
    if not message:
        return jsonify({ "error": "Missing 'message' in request body" }), 400

    context = retrieve(message)

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct:fastest",
            "messages": [
                { "role": "system", "content": f"You are a financial fraud consultant. Use this context:\n\n{context}" },
                { "role": "user", "content": message }
            ]
        }
    )

    return jsonify(response.json())

if __name__ == "__main__":
    app.run(port=3002)