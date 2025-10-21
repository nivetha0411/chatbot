# app.py
import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__, static_folder="static")

# Enable CORS (important if frontend served separately)
from flask_cors import CORS
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json() or {}
    message = payload.get("message", "")
    history = payload.get("history", [])  # array of {role:, content:}
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Build messages array for the model
    messages = history + [{"role": "user", "content": message}]

    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": float(os.getenv("TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("MAX_TOKENS", 800))
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Debug print to see full API response in logs
        print("RAW RESPONSE:", data)

        # Safely extract assistant reply
        assistant_text = None
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                assistant_text = choice["message"]["content"]
            elif "text" in choice:
                assistant_text = choice["text"]
            elif "delta" in choice and "content" in choice["delta"]:
                assistant_text = choice["delta"]["content"]

        if not assistant_text:
            return jsonify({"error": "No valid reply from model", "raw": data}), 500

        return jsonify({"reply": assistant_text, "raw": data})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Upstream request failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
