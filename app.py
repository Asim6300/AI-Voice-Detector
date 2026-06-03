from flask import Flask, request, render_template, jsonify
from predict import predict
import os

app = Flask(__name__)

UPLOAD = "uploads"
os.makedirs(UPLOAD, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def detect():

    if "audio" not in request.files:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json:
            return jsonify({"error": "No file uploaded"}), 400
        return "No file uploaded"

    file = request.files["audio"]

    if file.filename == "":
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json:
            return jsonify({"error": "No file selected"}), 400
        return "No file selected"

    path = os.path.join(
        UPLOAD,
        file.filename
    )

    file.save(path)

    try:
        result = predict(path)
    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json:
            return jsonify({"error": str(e)}), 500
        return f"Error predicting: {str(e)}"

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.accept_json:
        return jsonify(result)

    return f"""
    <h2>Result: {result['result']}</h2>

    <h3>Human: {result['human']}%</h3>

    <h3>AI: {result['ai']}%</h3>

    <a href="/">Try another</a>
    """

if __name__=="__main__":
    app.run(debug=True)