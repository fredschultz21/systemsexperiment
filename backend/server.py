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
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://systemsexperiment.vercel.app"])

# Lazy loading — imports torch only on first request
_model = None
_supabase = None

def get_clients():
    global _model, _supabase
    if _model is None:
        logger.info("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        from supabase import create_client
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )
        logger.info("Ready")
    return _model, _supabase

def retrieve(query: str, top_k: int = 5) -> str:
    try:
        model, supabase = get_clients()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()
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

def call_model_api(prompt, hf_url):
    prompt_formats = [
        f"Question: {prompt}\n\nAnswer:",
        f"Please answer this question about fraud prevention: {prompt}",
        f"Human: {prompt}\nAssistant:",
        prompt
    ]
    headers = {"Content-Type": "application/json"}
    for i, formatted_prompt in enumerate(prompt_formats):
        try:
            response = requests.post(
                f"{hf_url}/generate",
                headers=headers,
                json={"inputs": formatted_prompt},
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "").strip()
                    if generated_text and is_valid_and_complete_text(generated_text):
                        return generated_text
            logger.warning(f"Format {i+1} failed: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Format {i+1} failed: {e}")
    raise Exception("All prompt formats failed")

def is_valid_and_complete_text(text):
    if not text or len(text) < 10:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    garbled_chars = 'xjqzv'
    garbled_score = sum(1 for char in text.lower() if char in garbled_chars)
    if len(text) and garbled_score / len(text) > 0.3:
        return False
    unique_words = set(words[:15])
    if len(unique_words) < len(words[:15]) * 0.4:
        return False
    if len(text) < 20 and not text.rstrip().endswith(('.', '!', '?', ':', ';')):
        return False
    if text.endswith((' how', ' what', ' the', ' and', ' but', ' or', ' to', ' can', ' will')):
        return False
    return True

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "version": "4.0"})

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
        hf_url = "https://fredschultz-qwen-lora-api.hf.space:8000"
        ai_response = call_model_api(full_prompt, hf_url)
        if len(ai_response.split()) < 5:
            ai_response = "I can help with Iowa City housing and fraud prevention. Could you be more specific?"
        return jsonify({
            "choices": [{"message": {"content": ai_response}}],
            "model": "qwen2-vl",
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e), "fallback_response": "Having trouble right now, please try again."}), 500

@app.route("/test", methods=["GET"])
def test_simple():
    return jsonify({"status": "OK", "rag": "Supabase vector search active"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)