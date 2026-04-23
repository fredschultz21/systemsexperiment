import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# ── Supabase + embedding model (loads once on startup) ───────────────────────
logger.info("Loading embedding model...")
_model    = SentenceTransformer("all-MiniLM-L6-v2")
_supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)
logger.info("Embedding model and Supabase client ready")

# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> str:
    """Embed query, search Supabase for similar chunks, return context string."""
    try:
        query_vector = _model.encode(query, normalize_embeddings=True).tolist()

        result = _supabase.rpc("match_chunks", {
            "query_embedding": query_vector,
            "match_count":     top_k,
        }).execute()

        if not result.data:
            return "No relevant context found."

        context_parts = []
        for chunk in result.data:
            source = chunk.get("source_name", "Unknown source")
            text   = chunk.get("text", "")
            context_parts.append(f"[{source}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return "Error retrieving context."

# ── Model call ───────────────────────────────────────────────────────────────
def call_model_api(prompt, ngrok_url):
    """Call the model API with robust error handling and multiple prompt formats."""
    prompt_formats = [
        f"Question: {prompt}\n\nAnswer:",
        f"Please answer this question about fraud prevention: {prompt}",
        f"Human: {prompt}\nAssistant:",
        prompt
    ]

    headers = {
        "ngrok-skip-browser-warning": "true",
        "Content-Type": "application/json"
    }

    for i, formatted_prompt in enumerate(prompt_formats):
        try:
            logger.info(f"Trying prompt format {i+1}")
            response = requests.post(
                f"{ngrok_url}/generate",
                headers=headers,
                json={"inputs": formatted_prompt},
                timeout=90
            )
            logger.info(f"Response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "").strip()
                    if generated_text and is_valid_and_complete_text(generated_text):
                        logger.info(f"Success with format {i+1}")
                        return generated_text
                    else:
                        logger.warning(f"Format {i+1} produced incomplete/invalid text")
                        continue
                else:
                    logger.warning(f"Format {i+1}: Unexpected response format")
                    continue
            else:
                logger.warning(f"Format {i+1}: HTTP {response.status_code}")
                continue

        except Exception as e:
            logger.error(f"Format {i+1} failed: {e}")
            continue

    raise Exception("All prompt formats failed to generate valid text")

def is_valid_and_complete_text(text):
    """Check if generated text looks valid and reasonably complete."""
    if not text or len(text) < 10:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    garbled_chars = 'xjqzv'
    garbled_score = sum(1 for char in text.lower() if char in garbled_chars)
    garbled_ratio = garbled_score / len(text) if text else 0
    if garbled_ratio > 0.3:
        return False
    unique_words = set(words[:15])
    if len(unique_words) < len(words[:15]) * 0.4:
        return False
    if len(text) < 20 and not text.rstrip().endswith(('.', '!', '?', ':', ';')):
        return False
    if text.endswith((' how', ' what', ' the', ' and', ' but', ' or', ' to', ' can', ' will')):
        return False
    return True

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Production server running",
        "rag":     "Supabase vector search",
        "version": "4.0-supabase"
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        message = data.get("message")
        if not message:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        logger.info(f"Chat request: {message[:50]}...")

        # Retrieve relevant context from Supabase
        context = retrieve(message)
        logger.info(f"Context retrieved: {len(context)} chars")

        # Build prompt — allow up to 300 words of context
        if context and context != "No relevant context found." and len(context) > 50:
            context_words = context.split()[:300]
            limited_context = " ".join(context_words)
            full_prompt = f"""Answer this question: {message}

Use this information to help answer:
{limited_context}

Provide a complete, helpful answer:"""
        else:
            full_prompt = f"""Answer this question about housing and fraud prevention in Iowa City: {message}

Provide a complete, helpful answer:"""

        # Update this URL to your current ngrok or HuggingFace endpoint
        ngrok_url = "https://fredschultz-qwen-lora-api.hf.space:8000"

        ai_response = call_model_api(full_prompt, ngrok_url)

        if len(ai_response.split()) < 5:
            ai_response = "I can help with Iowa City housing and fraud prevention. Could you be more specific about what you'd like to know?"

        logger.info(f"Response generated: {len(ai_response)} chars")

        return jsonify({
            "choices": [{
                "message": {
                    "content": ai_response
                }
            }],
            "model": "local-model",
            "usage": {
                "prompt_tokens":      len(full_prompt.split()),
                "completion_tokens":  len(ai_response.split()),
                "total_tokens":       len(full_prompt.split()) + len(ai_response.split())
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": f"Failed to generate response: {str(e)}",
            "fallback_response": "I apologize, but I'm having trouble generating a response right now. Please try again."
        }), 500

@app.route("/test", methods=["GET"])
def test_simple():
    return jsonify({
        "status":    "OK",
        "message":   "Production server is running",
        "rag":       "Supabase vector search active",
        "endpoints": ["/", "/chat", "/test"]
    })

if __name__ == "__main__":
    logger.info("Starting PRODUCTION Flask server...")
    logger.info("RAG: Supabase vector search")
    logger.info("Server will run on: http://127.0.0.1:3002")
    app.run(
        host="0.0.0.0",
        port=3002,
        debug=False,
        use_reloader=False
    )