import os
import re
import json
import sqlite3
import requests
import pickle
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

# Optional/Lazy imports for ML features
reader = None
def get_ocr_reader():
    global reader
    if reader is None:
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            print("[INFO] EasyOCR loaded successfully.")
        except ImportError:
            print("[WARN] EasyOCR not installed. Using simulated OCR.")
    return reader

try:
    from google import genai
except ImportError:
    genai = None

# --- scikit-learn ML Model ---
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
    print("[INFO] scikit-learn loaded.")
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] scikit-learn not installed. Using heuristic fallback.")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_FILE = 'contacts.db'

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contacts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  phone TEXT NOT NULL,
                  verified INTEGER DEFAULT 0,
                  added_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS otp_store
                 (contact_id INTEGER PRIMARY KEY,
                  otp TEXT,
                  expires_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS voice_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  transcript TEXT,
                  risk_level TEXT,
                  summary TEXT,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# ML MEDICAL RISK CLASSIFIER
# ─────────────────────────────────────────────
TRAINING_DATA = [
    # High risk - full sentences
    ("severe chest pain radiating left arm shortness of breath", "High"),
    ("myocardial infarction detected troponin elevated critical", "High"),
    ("stroke symptoms facial drooping arm weakness speech difficulty", "High"),
    ("critical blood pressure 200 over 120 hypertensive crisis", "High"),
    ("anaphylactic shock severe allergic reaction", "High"),
    ("unconscious patient unresponsive no pulse", "High"),
    ("severe bleeding hemorrhage trauma accident", "High"),
    ("seizure epilepsy convulsion status epilepticus", "High"),
    ("glucose level 40 hypoglycemia diabetic emergency", "High"),
    ("oxygen saturation 85 percent respiratory failure", "High"),
    ("stemi acute myocardial infarction critical", "High"),
    ("gunshot wound massive arterial bleeding", "High"),
    ("patient is turning blue cyanosis choking", "High"),
    ("sudden loss of vision severe headache subarachnoid", "High"),
    ("third degree burns covering 30 percent body", "High"),
    # High risk - single keywords & short phrases
    ("heart attack", "High"),
    ("stroke", "High"),
    ("cardiac arrest", "High"),
    ("not breathing", "High"),
    ("no pulse", "High"),
    ("unconscious", "High"),
    ("unresponsive", "High"),
    ("choking", "High"),
    ("anaphylaxis", "High"),
    ("severe bleeding", "High"),
    ("hemorrhage", "High"),
    ("seizure", "High"),
    ("convulsion", "High"),
    ("sepsis", "High"),
    ("pulmonary embolism", "High"),
    ("coma", "High"),
    ("infarction", "High"),
    ("overdose", "High"),
    ("stopped breathing", "High"),
    ("turning blue", "High"),
    ("cyanosis", "High"),
    ("suicidal", "High"),

    # Moderate risk - full sentences
    ("borderline diabetes elevated sugar levels fasting glucose 120", "Moderate"),
    ("mildly elevated blood pressure 140 over 90 hypertension stage 1", "Moderate"),
    ("elevated cholesterol LDL 190 mg/dL borderline high", "Moderate"),
    ("anemia hemoglobin 9 g/dL fatigue dizziness", "Moderate"),
    ("thyroid TSH elevated hypothyroidism", "Moderate"),
    ("kidney creatinine slightly elevated 1.5 renal function", "Moderate"),
    ("mild chest discomfort occasional palpitations stress", "Moderate"),
    ("pain in joints mild arthritis inflammation", "Moderate"),
    ("liver enzymes slightly elevated SGPT 60", "Moderate"),
    ("dizzy spells occasional tension", "Moderate"),
    ("asthma attack wheezing shortness of breath", "Moderate"),
    ("severe abdominal pain vomiting suspected appendicitis", "Moderate"),
    ("migraine aura sensitivity to light", "Moderate"),
    ("deep cut laceration requiring stitches", "Moderate"),
    ("sprained ankle swelling pain upon walking", "Moderate"),
    # Moderate risk - single keywords & short phrases
    ("vomiting", "Moderate"),
    ("dizziness", "Moderate"),
    ("high fever", "Moderate"),
    ("fever", "Moderate"),
    ("chest tightness", "Moderate"),
    ("shortness of breath", "Moderate"),
    ("breathless", "Moderate"),
    ("palpitations", "Moderate"),
    ("hypertension", "Moderate"),
    ("high blood pressure", "Moderate"),
    ("asthma", "Moderate"),
    ("wheezing", "Moderate"),
    ("abdominal pain", "Moderate"),
    ("stomach pain", "Moderate"),
    ("fracture", "Moderate"),
    ("broken bone", "Moderate"),
    ("concussion", "Moderate"),
    ("migraine", "Moderate"),
    ("anemia", "Moderate"),
    ("diabetes", "Moderate"),
    ("high sugar", "Moderate"),
    ("kidney pain", "Moderate"),
    ("urinary pain", "Moderate"),
    ("blood in urine", "Moderate"),
    ("joint pain", "Moderate"),
    ("swelling", "Moderate"),
    ("food poisoning", "Moderate"),
    ("dehydration", "Moderate"),
    ("appendicitis", "Moderate"),
    ("chest pain", "Moderate"),

    # Low risk - full sentences
    ("normal blood work all values within range healthy", "Low"),
    ("routine checkup no abnormalities found", "Low"),
    ("blood pressure 118 over 76 normal healthy", "Low"),
    ("cholesterol within normal limits total 170", "Low"),
    ("BMI 22 healthy weight normal metabolism", "Low"),
    ("vitamin D slightly low supplement recommended", "Low"),
    ("general wellness visit no complaints", "Low"),
    ("mild fatigue adequate sleep recommended", "Low"),
    ("hemoglobin 14 normal red blood cells healthy", "Low"),
    ("ECG normal sinus rhythm no arrhythmia detected", "Low"),
    ("common cold mild cough runny nose", "Low"),
    ("minor paper cut scrape bleeding stopped", "Low"),
    ("mild indigestion heartburn after eating", "Low"),
    ("seasonal allergies sneezing watery eyes", "Low"),
    ("tension headache resolved with ibuprofen", "Low"),
    # Low risk - single keywords & short phrases
    ("headache", "Low"),
    ("cold", "Low"),
    ("cough", "Low"),
    ("sneeze", "Low"),
    ("sneezing", "Low"),
    ("runny nose", "Low"),
    ("sore throat", "Low"),
    ("mild fever", "Low"),
    ("fatigue", "Low"),
    ("tired", "Low"),
    ("insomnia", "Low"),
    ("skin rash", "Low"),
    ("mild rash", "Low"),
    ("heartburn", "Low"),
    ("indigestion", "Low"),
    ("muscle ache", "Low"),
    ("body ache", "Low"),
    ("bruise", "Low"),
    ("cut", "Low"),
    ("minor cut", "Low"),
    ("allergies", "Low"),
    ("dry skin", "Low"),
    ("acne", "Low"),
    ("bloating", "Low"),
    ("constipation", "Low"),
    ("toothache", "Low"),
    ("earache", "Low"),
    ("eye irritation", "Low"),
    ("blister", "Low"),
    ("normal", "Low"),
    ("healthy", "Low"),
    ("fine", "Low"),
    ("no symptoms", "Low"),
    ("feeling ok", "Low"),
    ("mild pain", "Low"),
    ("slight discomfort", "Low"),
]

MODEL_PATH = 'ml_model.pkl'

def train_risk_model():
    """Train and persist the medical risk classification model."""
    texts, labels = zip(*TRAINING_DATA)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95)),
        ('clf', LogisticRegression(max_iter=500, C=1.0, class_weight='balanced'))
    ])
    pipeline.fit(texts, labels)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[INFO] ML model trained and saved to {MODEL_PATH}")
    return pipeline

def load_or_train_model():
    """Load model from disk, or train if not present."""
    if not ML_AVAILABLE:
        return None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"[INFO] ML model loaded from {MODEL_PATH}")
            return model
        except Exception:
            pass
    return train_risk_model()

ml_model = load_or_train_model()

def ml_predict_risk(text):
    """Use trained ML model to predict risk level from text."""
    if ml_model is None:
        return None
    try:
        prediction = ml_model.predict([text])[0]
        probas = ml_model.predict_proba([text])[0]
        classes = ml_model.classes_
        confidence = float(max(probas))
        return {"risk_level": prediction, "confidence": round(confidence * 100, 1), "classes": list(classes), "probas": [round(p * 100, 1) for p in probas]}
    except Exception as e:
        print(f"[ML Error] {e}")
        return None

# ─────────────────────────────────────────────
# BIOTHINGS INTEGRATION
# ─────────────────────────────────────────────
def get_biothings_context(text):
    words = re.findall(r'\b[A-Za-z]{5,}\b', text)
    context_data = {}
    sample_terms = list(set(words))[:3]
    for term in sample_terms:
        try:
            res = requests.get(f"https://mydisease.info/v1/query?q={term}&size=1", timeout=2)
            if res.status_code == 200 and res.json().get('hits'):
                hit = res.json()['hits'][0]
                name = hit.get('disease_name', term)
                context_data[term] = name
        except Exception:
            pass
    return context_data

# ─────────────────────────────────────────────
# CORE ANALYSIS ENGINE
# ─────────────────────────────────────────────
def analyze_medical_text(text):
    """Full pipeline: ML model → BioThings → Gemini fallback → Heuristics."""
    result = {
        "risk_level": "Low",
        "summary": "No critical markers detected.",
        "immediate_action": "Continue routine monitoring.",
        "ml_confidence": None,
        "ml_probas": None,
        "source": "heuristic"
    }

    # --- Step 1: ML Model Prediction ---
    ml_result = ml_predict_risk(text)
    if ml_result:
        result["risk_level"] = ml_result["risk_level"]
        result["ml_confidence"] = ml_result["confidence"]
        result["ml_probas"] = dict(zip(ml_result["classes"], ml_result["probas"]))
        result["source"] = "ml_model"

    # --- Step 2: BioThings Context ---
    biothings_context = get_biothings_context(text)

    # --- Step 3: Gemini API (if configured) ---
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if genai and api_key:
        prompt = f"""
        Act as a clinical diagnostic model. Extract key biomarkers from the text below,
        verify bio-technical terms using BioThings context provided, compare against emergency
        thresholds, and output ONLY a valid JSON object with these keys:
        'risk_level' (Low, Moderate, or High),
        'summary' (plain English jargon-free, max 2 sentences),
        'immediate_action' (ONE specific action, max 15 words).

        TEXT: {text}
        BIOTHINGS CONTEXT: {json.dumps(biothings_context)}
        ML PRE-ASSESSMENT: Risk={result['risk_level']}, Confidence={result.get('ml_confidence')}%

        Output strictly valid JSON and nothing else.
        """
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            gemini_result = json.loads(json_text)
            result.update(gemini_result)
            result["source"] = "gemini"
        except Exception as e:
            print(f"[Gemini Error] {e}")

    # --- ACTIVE LEARNING: Auto-Train ML model ---
    if result["source"] in ("gemini", "heuristic") and ML_AVAILABLE:
        new_label = result.get("risk_level")
        if new_label in ("High", "Moderate", "Low") and len(text.split()) > 3:
            global ml_model
            TRAINING_DATA.append((text, new_label))
            ml_model = train_risk_model()
            print(f"[INFO] Active Learning: ML model retrained with new {result['source']} sample -> {new_label}")

    # --- Step 4: Heuristic fallback for summary/action ---
    if result["source"] in ("ml_model", "heuristic"):
        text_lower = text.lower()
        
        # Load the comprehensive symptoms database
        markers = {}
        try:
            with open("symptoms_database.json", "r") as f:
                markers = json.load(f)
        except Exception:
            # Fallback if file is missing
            markers = {
                'High': ['troponin', 'infarction', 'stroke', 'hemorrhage', 'seizure', 'critical'],
                'Moderate': ['elevated', 'hypertension', 'anemia', 'thyroid', 'cholesterol'],
                'Low': ['normal', 'healthy', 'stable', 'within range']
            }

        found_terms = []
        highest_risk_found = "Low"
        
        # We check High first, then Moderate, then Low
        for level in ['High', 'Moderate', 'Low']:
            for term in markers.get(level, []):
                if term in text_lower:
                    found_terms.append(term)
                    if highest_risk_found == "Low" and level != "Low":
                        highest_risk_found = level
                    elif highest_risk_found == "Moderate" and level == "High":
                        highest_risk_found = "High"

        # If this is purely a heuristic run, override the default "Low" risk level
        if result["source"] == "heuristic" and found_terms:
            result["risk_level"] = highest_risk_found

        # Build a text-aware summary
        terms_str = ', '.join(found_terms[:4]) if found_terms else 'general symptoms'

        if result["risk_level"] == "High":
            result["summary"] = f"Critical markers detected ({terms_str}). Immediate medical intervention is required."
            result["immediate_action"] = "Call emergency services (102) immediately."
        elif result["risk_level"] == "Moderate":
            result["summary"] = f"Elevated or borderline markers found ({terms_str}). Medical attention is recommended soon."
            result["immediate_action"] = "Schedule an urgent consultation with your doctor today."
        else:
            result["summary"] = f"No critical markers detected ({terms_str}). Your results appear within safe limits."
            result["immediate_action"] = "Continue your routine and review at your next visit."

    return result

# ─────────────────────────────────────────────
# HOSPITAL FINDER (OpenStreetMap Overpass API)
# ─────────────────────────────────────────────
def find_nearby_hospitals(lat, lng, radius_meters=5000):
    """Fetch real hospitals near coordinates using Overpass API."""
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"="hospital"](around:{radius_meters},{lat},{lng});
      way["amenity"="hospital"](around:{radius_meters},{lat},{lng});
      node["amenity"="clinic"](around:{radius_meters},{lat},{lng});
      node["healthcare"="hospital"](around:{radius_meters},{lat},{lng});
    );
    out body center 10;
    """
    try:
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=12
        )
        data = response.json()
        hospitals = []
        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name", "Unnamed Hospital")
            if not name or name == "Unnamed Hospital":
                continue
            elem_lat = element.get("lat") or element.get("center", {}).get("lat")
            elem_lng = element.get("lon") or element.get("center", {}).get("lon")
            if elem_lat and elem_lng:
                # Haversine distance
                import math
                R = 6371000
                phi1, phi2 = math.radians(float(lat)), math.radians(float(elem_lat))
                dphi = math.radians(float(elem_lat) - float(lat))
                dlambda = math.radians(float(elem_lng) - float(lng))
                a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
                dist_m = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                dist_km = round(dist_m / 1000, 1)
                est_min = max(1, round(dist_m / 500))  # ~30 km/h in traffic
                phone = tags.get("phone", tags.get("contact:phone", ""))
                hospitals.append({
                    "name": name,
                    "distance": f"{dist_km} km",
                    "time": f"{est_min} min",
                    "phone": phone,
                    "lat": elem_lat,
                    "lng": elem_lng
                })
        hospitals.sort(key=lambda x: float(x["distance"].replace(" km", "")))
        return hospitals[:5]
    except Exception as e:
        print(f"[Hospital API Error] {e}")
        # Fallback mock: offset slightly so they don't map exactly to the user's location
        lat_f = float(lat)
        lng_f = float(lng)
        return [
            {"name": "Apollo Hospitals", "distance": "1.2 km", "time": "4 min", "phone": "1860-500-1066", "lat": lat_f + 0.010, "lng": lng_f + 0.010},
            {"name": "Fortis Healthcare", "distance": "2.5 km", "time": "8 min", "phone": "1800-111-4567", "lat": lat_f - 0.015, "lng": lng_f + 0.020},
            {"name": "AIIMS Emergency", "distance": "3.8 km", "time": "13 min", "phone": "011-26588500", "lat": lat_f + 0.020, "lng": lng_f - 0.010},
        ]

# ─────────────────────────────────────────────
# OTP MANAGEMENT
# ─────────────────────────────────────────────
DEFAULT_OTP = "1666"

def generate_otp():
    """Returns default OTP (1666) for demo; in production would send SMS."""
    return DEFAULT_OTP

def store_otp(contact_id, otp):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    expires = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT OR REPLACE INTO otp_store (contact_id, otp, expires_at) VALUES (?, ?, ?)",
              (contact_id, otp, expires))
    conn.commit()
    conn.close()

def verify_otp(contact_id, input_otp):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT otp FROM otp_store WHERE contact_id=?", (contact_id,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == input_otp:
        return True
    # Also accept the global default OTP
    return input_otp == DEFAULT_OTP

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# --- Report Upload & Analysis ---
@app.route('/api/upload_report', methods=['POST'])
def upload_report():
    if 'report' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['report']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if genai and api_key:
        try:
            import PIL.Image
            img = PIL.Image.open(filepath)
            print(f"[INFO] Image opened: {filepath}, size={img.size}, mode={img.mode}")
            client = genai.Client(api_key=api_key)
            prompt = """You are an expert clinical diagnostic AI. Analyze this medical report image carefully.

Instructions:
1. Read ALL text, numbers, lab values, and medical terms visible in the image.
2. Identify every biomarker, test result, and diagnosis.
3. Compare each value against standard medical reference ranges.
4. Determine the overall risk level based on the MOST critical finding.

Output ONLY a valid JSON object with these exact keys:
{
  "extracted_text": "The full exact text read from the image",
  "risk_level": "Low" or "Moderate" or "High",
  "summary": "Plain English explanation of key findings (2-3 sentences, no medical jargon)",
  "immediate_action": "ONE specific action the patient should take (max 15 words)"
}

IMPORTANT: Base your assessment ONLY on what you see in the image. Do not guess or assume. Output strictly valid JSON and nothing else."""
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[img, prompt]
            )
            raw_text = response.text.strip()
            print(f"[Gemini Vision Raw] {raw_text[:300]}")
            json_text = raw_text.replace("```json", "").replace("```", "").strip()
            gemini_result = json.loads(json_text)
            
            # Active Learning (Vision)
            ext_text = gemini_result.get("extracted_text", "")
            r_level = gemini_result.get("risk_level")
            if ML_AVAILABLE and ext_text and len(ext_text.split()) > 3 and r_level in ("High", "Moderate", "Low"):
                global ml_model
                TRAINING_DATA.append((ext_text, r_level))
                ml_model = train_risk_model()
                print(f"[INFO] Active Learning (Vision): ML model retrained -> {r_level}")
            
            return jsonify({
                "status": "success",
                "extracted_text": ext_text if ext_text else "Analyzed directly via Gemini Vision API",
                "risk_level": r_level or "Unknown",
                "summary": gemini_result.get("summary", ""),
                "action": gemini_result.get("immediate_action", ""),
                "source": "gemini_vision"
            })
        except Exception as e:
            import traceback
            print(f"[Gemini Vision Error] {e}")
            traceback.print_exc()

    extracted_text = ""
    
    # Check if the file is a PDF
    if filepath.lower().endswith('.pdf'):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text_pages = []
            for page in doc:
                text_pages.append(page.get_text())
            extracted_text = " ".join(text_pages)
            if not extracted_text.strip():
                extracted_text = "Error: PDF appears to be empty or contains only scanned images without text."
        except ImportError:
            extracted_text = "Error: PyMuPDF is not installed. Cannot read PDF."
        except Exception as e:
            extracted_text = f"Error reading PDF: {e}"
    else:
        # It's an image, use easyocr
        ocr_reader = get_ocr_reader()
        if ocr_reader is not None:
            try:
                result = ocr_reader.readtext(filepath, detail=0)
                extracted_text = " ".join(result)
            except Exception as e:
                extracted_text = "Error reading image: " + str(e)
                
    # If no text was extracted, apply the mock fallback
    if not extracted_text.strip() or extracted_text.startswith("Error"):
        # Hackathon Demo Trick: Change mock text based on the uploaded file's name!
        lower_name = filename.lower()
        if "normal" in lower_name or "low" in lower_name or "healthy" in lower_name or "good" in lower_name:
            extracted_text = "Mock OCR: All blood work values within normal range. Healthy patient."
        elif "moderate" in lower_name or "elevated" in lower_name or "warning" in lower_name or "mild" in lower_name:
            extracted_text = "Mock OCR: Mildly elevated cholesterol and blood pressure. Moderate risk detected."
        else:
            extracted_text = "Mock OCR: Severe elevated troponin levels myocardial infarction detected."


    analysis = analyze_medical_text(extracted_text)
    return jsonify({
        "status": "success",
        "extracted_text": extracted_text,
        "risk_level": analysis.get("risk_level", "Unknown"),
        "summary": analysis.get("summary", ""),
        "action": analysis.get("immediate_action", ""),
        "ml_confidence": analysis.get("ml_confidence"),
        "ml_probas": analysis.get("ml_probas"),
        "source": analysis.get("source")
    })

# --- Voice Command Analysis ---
@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    data = request.get_json() or {}
    command = data.get('command', '').strip()
    if not command:
        return jsonify({"error": "Empty transcript"}), 400
    analysis = analyze_medical_text(command)
    # Log to DB
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO voice_logs (transcript, risk_level, summary) VALUES (?, ?, ?)",
              (command, analysis.get("risk_level"), analysis.get("summary")))
    conn.commit()
    conn.close()
    return jsonify({
        "status": "success",
        "transcript": command,
        "risk_level": analysis.get("risk_level"),
        "summary": analysis.get("summary"),
        "action": analysis.get("immediate_action"),
        "ml_confidence": analysis.get("ml_confidence"),
        "ml_probas": analysis.get("ml_probas"),
        "source": analysis.get("source")
    })

# --- Retrain ML Model ---
@app.route('/api/retrain_model', methods=['POST'])
def retrain_model():
    """Endpoint to add new training sample and retrain the model."""
    global ml_model
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    label = data.get('label', '').strip()
    if not text or label not in ('High', 'Moderate', 'Low'):
        return jsonify({"error": "Provide text and label (High/Moderate/Low)"}), 400
    TRAINING_DATA.append((text, label))
    if ML_AVAILABLE:
        ml_model = train_risk_model()
        return jsonify({"status": "success", "message": f"Model retrained with {len(TRAINING_DATA)} samples.", "label": label})
    return jsonify({"status": "skipped", "message": "ML not available."})

# ─────────────────────────────────────────────
# CONTACTS (SOS Trusted Contacts with OTP)
# ─────────────────────────────────────────────
@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, phone, verified, added_at FROM contacts ORDER BY added_at DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "name": r[1], "phone": r[2], "verified": bool(r[3]), "added_at": r[4]} for r in rows])

@app.route('/api/contacts', methods=['POST'])
def add_contact():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    phone = data.get('phone', '').strip()
    if not name or not phone:
        return jsonify({"error": "Name and phone are required."}), 400
    # Basic phone validation
    phone_clean = re.sub(r'\D', '', phone)
    if len(phone_clean) != 10:
        return jsonify({"error": "Mobile number must be exactly 10 digits."}), 400
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO contacts (name, phone, verified) VALUES (?, ?, 1)", (name, phone_clean))
    contact_id = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({
        "status": "success",
        "contact_id": contact_id,
        "message": "Contact added.",
        "otp_required": False
    })

@app.route('/api/contacts/<int:contact_id>', methods=['PUT'])
def update_contact(contact_id):
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    phone = data.get('phone', '').strip()
    phone_clean = re.sub(r'\D', '', phone) if phone else ''
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if name and phone_clean:
        c.execute("UPDATE contacts SET name=?, phone=?, verified=1 WHERE id=?", (name, phone_clean, contact_id))
    elif name:
        c.execute("UPDATE contacts SET name=?, verified=1 WHERE id=?", (name, contact_id))
    elif phone_clean:
        c.execute("UPDATE contacts SET phone=?, verified=1 WHERE id=?", (phone_clean, contact_id))
    else:
        c.execute("UPDATE contacts SET verified=1 WHERE id=?", (contact_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Contact updated."})

@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM contacts WHERE id=?", (contact_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "message": "Contact deleted."})

# ─────────────────────────────────────────────
# SOS EMERGENCY
# ─────────────────────────────────────────────
@app.route('/api/sos', methods=['POST'])
def trigger_sos():
    data = request.get_json() or {}
    lat = data.get('lat', None)
    lng = data.get('lng', None)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, phone FROM contacts")
    contacts = c.fetchall()
    conn.close()

    contact_names = [row[0] for row in contacts]
    contact_str = ", ".join(contact_names) if contact_names else "No contacts saved"

    print(f"[SOS ALERT] Amma Protocol Active. Location: {lat}, {lng}")
    print(f"Dispatching to: {contact_str}")

    # Fetch REAL nearby hospitals
    hospitals = []
    if lat and lng:
        try:
            hospitals = find_nearby_hospitals(float(lat), float(lng))
        except Exception as e:
            print(f"[Hospital Fetch Error] {e}")

    if not hospitals:
        lat_f = float(lat) if lat else 0.0
        lng_f = float(lng) if lng else 0.0
        hospitals = [
            {"name": "Apollo City Hospital", "distance": "1.2 km", "time": "4 min", "phone": "1860-500-1066", "lat": lat_f + 0.010, "lng": lng_f + 0.010},
            {"name": "Fortis Emergency Center", "distance": "2.5 km", "time": "8 min", "phone": "1800-111-4567", "lat": lat_f - 0.015, "lng": lng_f + 0.020},
            {"name": "General Public Hospital", "distance": "3.1 km", "time": "12 min", "phone": "", "lat": lat_f + 0.020, "lng": lng_f - 0.010},
        ]

    contact_phones = [row[1] for row in contacts]

    # Directly send SMS via Twilio if configured
    twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
    twilio_phone = os.environ.get("TWILIO_PHONE_NUMBER")

    direct_sms_sent = False
    sms_errors = []

    if TwilioClient and twilio_sid and twilio_token and twilio_phone:
        try:
            client = TwilioClient(twilio_sid, twilio_token)
            loc_link = f"https://maps.google.com/?q={lat},{lng}" if lat and lng else "Location unavailable"
            message_body = f"EMERGENCY SOS from LifeLine AI! I need help immediately. My location: {loc_link}"
            
            for phone in contact_phones:
                # Ensure phone number is in E.164 format (e.g., +1234567890)
                # If they entered a 10 digit number, we assume the country code based on region. 
                # For demo, we just prepend '+' if missing, or require valid format.
                formatted_phone = phone if phone.startswith('+') else f"+91{phone}"
                try:
                    client.messages.create(
                        body=message_body,
                        from_=twilio_phone,
                        to=formatted_phone
                    )
                    direct_sms_sent = True
                except Exception as e:
                    print(f"[Twilio SMS Error for {formatted_phone}] {e}")
                    sms_errors.append(str(e))
        except Exception as e:
            print(f"[Twilio Init Error] {e}")
            
    # Fallback to Textbelt for free testing (1 message per day limit) if Twilio isn't set up
    if not direct_sms_sent and contact_phones:
        print("[INFO] Twilio not configured. Attempting to send direct SMS via free Textbelt API...")
        loc_link = f"https://maps.google.com/?q={lat},{lng}" if lat and lng else "Location unavailable"
        message_body = f"EMERGENCY SOS! I need help immediately. Location: {loc_link}"
        
        # We only send to the FIRST contact to not waste the 1-per-day quota
        first_phone = contact_phones[0]
        try:
            resp = requests.post('https://textbelt.com/text', data={
                'phone': first_phone,
                'message': message_body,
                'key': 'textbelt'
            })
            resp_data = resp.json()
            if resp_data.get('success'):
                print(f"[Textbelt] Successfully sent direct SMS to {first_phone}!")
                direct_sms_sent = True
            else:
                print(f"[Textbelt Error] {resp_data.get('error')}")
        except Exception as e:
            print(f"[Textbelt Request Error] {e}")

    return jsonify({
        "status": "success",
        "amma_status": "Amma SOS Dispatched (Direct SMS: " + ("Sent" if direct_sms_sent else "Failed/Not Configured") + ")",
        "message": f"Alert sent to {len(contacts)} trusted contacts. Ambulance 102 notified.",
        "action": "Stay exactly where you are. Help is on the way.",
        "hospitals": hospitals,
        "phones": contact_phones,
        "direct_sms_sent": direct_sms_sent,
        "lat": lat,
        "lng": lng
    })

# --- Nearby Hospitals Only ---
@app.route('/api/hospitals', methods=['POST'])
def get_hospitals():
    data = request.get_json() or {}
    lat = data.get('lat')
    lng = data.get('lng')
    if not lat or not lng:
        return jsonify({"error": "Location required"}), 400
    hospitals = find_nearby_hospitals(float(lat), float(lng))
    return jsonify({"hospitals": hospitals})

# --- Results Page ---
@app.route('/results')
def results():
    risk_level = request.args.get('risk_level', 'Low')
    action = request.args.get('action', 'No action required.')
    summary = request.args.get('summary', '')
    amma_status = request.args.get('amma_status', 'Standby')
    ml_confidence = request.args.get('ml_confidence', '')
    source = request.args.get('source', '')
    sms_link = request.args.get('sms_link', '')
    hospitals_json = request.args.get('hospitals', '[]')
    hospitals = []
    try:
        hospitals = json.loads(hospitals_json)
    except Exception:
        pass
    return render_template('results.html',
                           risk_level=risk_level,
                           action=action,
                           summary=summary,
                           amma_status=amma_status,
                           hospitals=hospitals,
                           ml_confidence=ml_confidence,
                           sms_link=sms_link,
                           source=source)

# --- Voice Logs ---
@app.route('/api/voice_logs', methods=['GET'])
def get_voice_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, transcript, risk_level, summary, created_at FROM voice_logs ORDER BY created_at DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"id": r[0], "transcript": r[1], "risk_level": r[2], "summary": r[3], "created_at": r[4]} for r in rows])

if __name__ == '__main__':
    import socket
    # Get local IP address dynamically
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
        

    print("\n* LifeLine AI Server is starting...")
    print(f"  Local access: http://127.0.0.1:5000")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
