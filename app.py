#pip install -r requirements.txt
#spaCy version: 3.8.4
#Flask version: 3.0.3
from flask_cors import CORS
import spacy 
import re
from flask import Flask, request, jsonify 

app = Flask(__name__)
CORS(app)  # This allows all domains; you can restrict if needed
nlp = spacy.load("en_core_web_trf")  # Transformer-based model for accurate NER

# Improved regex for structured address detection
ADDRESS_PATTERN = r'(\d{1,5}[A-Za-z]?\s*[A-Za-z0-9\s,-]+(?:Road|Street|Block|Town|Industrial Estate|Markaz|Near|Phase|Sector|Chowk|Plaza).*\b(?:Lahore|Karachi|Islamabad|Sialkot)\b)'

# List of valid Pakistani cities for filtering
VALID_CITIES = {"Lahore", "Karachi", "Islamabad", "Sialkot"}

def extract_addresses_from_text(text):
    """Extract addresses using regex pattern."""
    return set(match.strip() for match in re.findall(ADDRESS_PATTERN, text, re.IGNORECASE))

def extract_locations_using_ner(text):
    """Extract city-specific locations using spaCy's NER and filter out unrelated places."""
    doc = nlp(text)
    extracted_locations = set()

    for ent in doc.ents:
        if ent.label_ == "GPE":  # Geopolitical Entity
            location = ent.text.title()  # Capitalize properly
            if location in VALID_CITIES:  # Only consider relevant cities
                extracted_locations.add(location)

    return extracted_locations

@app.route('/extract_addresses', methods=['POST'])
def extract_addresses():
    """API endpoint to extract addresses from text."""
    data = request.json
    text = data.get("text", "")
    
    # Extract addresses using regex
    addresses = extract_addresses_from_text(text)
    
    # Extract locations using NER
    locations = extract_locations_using_ner(text)

    # Merge results
    extracted_data = addresses | locations  # Union of both sets
    return jsonify(list(extracted_data))

if __name__ == '__main__':
    app.run(port=5000)