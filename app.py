import spacy
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load spaCy model for NER (Named Entity Recognition)
nlp = spacy.load("en_core_web_sm")

# Function to extract addresses using spaCy
def extract_addresses_spacy(text):
    doc = nlp(text)
    addresses = []
    for ent in doc.ents:
        if ent.label_ == "GPE" or ent.label_ == "LOC":
            addresses.append(ent.text)
    return addresses

# Flask route to extract addresses
@app.route("/extract-addresses", methods=["POST"])
def extract_addresses():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Extract addresses using both models
    addresses_spacy = extract_addresses_spacy(text)

    # Combine and remove duplicates
    extracted_data = list(set(addresses_spacy))

    return jsonify(extracted_data)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
