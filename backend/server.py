import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://systemsexperiment.vercel.app",
    "https://systemsexperiment-git-main-fredschultz21s-projects.vercel.app"
])

_supabase = None
_gradio_client = None

def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )
        logger.info("Supabase client ready")
    return _supabase

def get_gradio_client():
    global _gradio_client
    if _gradio_client is None:
        from gradio_client import Client
        _gradio_client = Client("fredschultz/qwen-lora-api")
        logger.info("Gradio client ready")
    return _gradio_client

def embed(text: str) -> list:
    response = requests.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
        json={"inputs": text},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return result[0]
    return result

def retrieve(query: str, top_k: int = 5) -> str:
    try:
        query_vector = embed(query)
        supabase = get_supabase()

        result = supabase.rpc("match_chunks", {
            "query_embedding": query_vector,
            "match_count": top_k,
        }).execute()

        if not result.data:
            return "No relevant context found."

        context_parts = []
        for chunk in result.data:
            source = chunk.get("source_name", "Unknown source")
            text = chunk.get("text", "")
            context_parts.append(f"[{source}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return "Error retrieving context."

def call_model(prompt: str) -> str:
    client = get_gradio_client()
    result = client.predict(
        prompt=prompt,
        api_name="/run_inference"
    )
    return result

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "version": "6.0"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        message = data.get("message")
        if not message:
            return jsonify({"error": "Missing message"}), 400

        context = retrieve(message)

        if context and context != "No relevant context found." and len(context) > 50:
            limited_context = " ".join(context.split()[:300])
            full_prompt = f"Answer this question: {message}\n\nUse this information:\n{limited_context}\n\nProvide a complete, helpful answer:"
        else:
            full_prompt = f"Answer this question about housing and fraud prevention in Iowa City: {message}\n\nProvide a complete, helpful answer:"

        ai_response = call_model(full_prompt)
        if "assistant\n" in ai_response:
            ai_response = ai_response.split("assistant\n")[-1].strip()

        if not ai_response or len(ai_response.split()) < 5:
            ai_response = "I can help with Iowa City housing and fraud prevention. Could you be more specific?"

        return jsonify({
            "choices": [{"message": {"content": ai_response}}],
            "model": "qwen2-vl",
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": str(e),
            "fallback_response": "Having trouble right now, please try again."
        }), 500

@app.route("/test", methods=["GET"])
def test_simple():
    return jsonify({"status": "OK", "rag": "Supabase vector search active"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)