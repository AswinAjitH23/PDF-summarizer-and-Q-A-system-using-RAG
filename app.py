from flask import Flask, render_template, request, jsonify
from rag_engine import get_rag_response

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"response": "Please enter a valid question."}), 400

    try:
        # Call RAG pipeline from rag_engine module
        bot_response = get_rag_response(user_query)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    print("\nStarting RAG Web Server on http://0.0.0.0:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)