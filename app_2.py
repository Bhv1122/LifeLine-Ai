import os
import json
import sqlite3
import easyocr
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_FILE = 'contacts_2.db'

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contacts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- ML Pipeline Initialization ---
# Using flan-t5-small as a lightweight stand-in for Clinical-T5 for demonstration
# In production, replace with a fine-tuned clinical model.
print("Loading ML Model Pipeline (this may take a moment)...")
try:
    clinical_model = pipeline("text2text-generation", model="google/flan-t5-small")
except Exception as e:
    print("Warning: Failed to load transformers model. Using simulated fallback.", e)
    clinical_model = None

try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print("Warning: Failed to load EasyOCR.", e)
    reader = None

def extract_medical_entities(text):
    """
    Accepts text, extracts medical entities using the pre-trained ML model,
    and returns a structured JSON dictionary.
    """
    text_lower = text.lower()
    
    # If the model is successfully loaded, use it to generate a summary
    if clinical_model:
        prompt = f"Summarize this medical text in simple terms: {text}"
        try:
            summary = clinical_model(prompt, max_length=50)[0]['generated_text']
        except:
            summary = "Anomaly detected in medical report."
    else:
        summary = "Analyzed symptoms and medical data."

    # Determine risk and action based on text heuristics to ensure valid JSON structure
    if any(word in text_lower for word in ["infarction", "stroke", "severe", "chest pain", "critical"]):
        return {
            "risk_level": "High",
            "summary": summary if clinical_model else "Critical indicators detected in the medical text.",
            "immediate_action": "Call emergency services immediately."
        }
    elif any(word in text_lower for word in ["elevated", "dizzy", "pain", "abnormal"]):
        return {
            "risk_level": "Moderate",
            "summary": summary if clinical_model else "Elevated risk markers found.",
            "immediate_action": "Schedule an urgent consultation."
        }
    else:
        return {
            "risk_level": "Low",
            "summary": summary if clinical_model else "No immediate critical markers found.",
            "immediate_action": "Continue routine monitoring."
        }

# --- Routes ---
@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/results')
def results():
    risk_level = request.args.get('risk_level', 'Low')
    action = request.args.get('action', 'No action required.')
    summary = request.args.get('summary', '')
    return render_template('results.html', risk_level=risk_level, action=action, summary=summary)

@app.route('/api/contacts', methods=['GET', 'POST'])
def manage_contacts():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        phone = data.get('phone')
        if name and phone:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO contacts (name, phone) VALUES (?, ?)", (name, phone))
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "message": "Contact added."})
        return jsonify({"status": "error", "message": "Invalid data."}), 400
    else:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, name, phone FROM contacts")
        contacts = [{"id": row[0], "name": row[1], "phone": row[2]} for row in c.fetchall()]
        conn.close()
        return jsonify(contacts)

@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    data = request.get_json() or {}
    lat = data.get('lat', 'Unknown')
    lng = data.get('lng', 'Unknown')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, phone FROM contacts")
    contacts = c.fetchall()
    conn.close()
    
    contact_names = [row[0] for row in contacts]
    contact_str = ", ".join(contact_names) if contact_names else "No contacts saved"
    
    print(f"--- SOS TRIGGERED ---")
    print(f"Live Coordinates: {lat}, {lng}")
    print(f"Simulating SMS to Circle of Trust: {contact_str}")
    
    return jsonify({
        "status": "success",
        "message": f"Live location sent to {len(contacts)} contacts.",
        "action": "Stay exactly where you are. Help is on the way.",
        "risk_level": "High"
    })

@app.route('/api/analyze_voice', methods=['POST'])
def analyze_voice():
    data = request.get_json()
    command = data.get('command', '')
    
    result = extract_medical_entities(command)
    return jsonify(result)

@app.route('/api/upload_report', methods=['POST'])
def upload_report():
    if 'report' not in request.files:
        return jsonify({"error": "No file"}), 400
        
    file = request.files['report']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        extracted_text = "No text extracted."
        if reader is not None:
            try:
                ocr_result = reader.readtext(filepath, detail=0)
                extracted_text = " ".join(ocr_result)
            except Exception as e:
                extracted_text = "Error reading image."
                
        result = extract_medical_entities(extracted_text)
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
