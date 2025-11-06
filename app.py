#=============================================================================================================
# Mark 1.1 ---- 
#=============================================================================================================


import os
import io
import json
import re
import base64
import time
import textwrap
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st
from streamlit_option_menu import option_menu

# imaging & utilities
from PIL import Image
import requests
import random

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# dotenv for env management
from dotenv import load_dotenv

# password hashing
import bcrypt

# sklearn for AI Assistant retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: Groq client (if installed and key present)
try:
    from groq import Groq
except Exception:
    Groq = None

# Optional barcode decode libraries
try:
    from pyzbar.pyzbar import decode as zbar_decode
    import numpy as np
    import cv2
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False

# Optional Lottie
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# -------------------------
# CONFIG & CONSTANTS
# -------------------------
st.set_page_config(page_title="Smart Nutrition Analyzer", page_icon="🍎", layout="wide")
load_dotenv()

ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
LOGO_PATH = "src/logo.png"
HISTORY_FILE = "report_history.json"
USERS_FILE = "users.json"   # stores registered users (email -> data)
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

MODEL_SETTINGS = {
    "temperature": float(os.getenv("MODEL_TEMPERATURE", 0.2)),
    "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", 400)),
    "top_p": float(os.getenv("MODEL_TOP_P", 0.5)),
}

# -------------------------
# HELPERS: safe rerun / init
# -------------------------
def safe_rerun():
    # compatibility for different streamlit versions
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# initialize session_state default keys to avoid AttributeError
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {}
if "ai_chat" not in st.session_state:
    st.session_state["ai_chat"] = []



# -------------------------
# Stylish Red-White Buttons
# -------------------------
st.markdown(
    """
    <style>
    /* General stylish buttons across app */
    div.stButton > button:first-child {
        background-color: #E53935;  /* red background */
        color: white;               /* white text */
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 24px;
        box-shadow: 0px 4px 12px rgba(229,57,53,0.3);
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:first-child:hover {
        background-color: #FF5252;  /* lighter red */
        transform: scale(1.05);
    }

    /* File uploader “Browse files” button */
    section[data-testid="stFileUploader"] label div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #E53935;
        background-color: rgba(255, 0, 0, 0.05);
        color: #E53935;
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"] label div[data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(255, 0, 0, 0.1);
        border-color: #FF5252;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# LOGO helper
# -------------------------
@st.cache_data
def get_logo_b64() -> Optional[str]:
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

# -------------------------
# USERS: load/save
# -------------------------
def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users: dict):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        st.error(f"Could not save users: {e}")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

# -------------------------
# HISTORY persistence
# -------------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Could not save history: {e}")

def add_history(entry: dict):
    h = load_history()
    h.insert(0, entry)
    save_history(h)

def clear_history_file():
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        st.error(f"Could not clear history: {e}")

# -------------------------
# PDF generation
# -------------------------
def generate_pdf_from_entry(title: str, text: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    header = ParagraphStyle("Header", parent=styles["Title"], alignment=1, fontSize=16, textColor=colors.white)
    hdr_tbl = Table([[Paragraph(f"<b>{title}</b>", header)]], colWidths=[450])
    hdr_tbl.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#2e7d32")), ("ALIGN",(0,0),(-1,-1),"CENTER")]))
    story.append(hdr_tbl)
    story.append(Spacer(1,10))
    for line in str(text).splitlines():
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1,6))
    try:
        doc.build(story)
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
    buf.seek(0)
    return buf

# -------------------------
# IMAGE helpers
# -------------------------
def pil_image_to_b64_and_fmt(pil_image: Image.Image) -> Tuple[str, str]:
    buf = io.BytesIO()
    fmt = pil_image.format or "PNG"
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), fmt

# -------------------------
# Groq client init (optional)
# -------------------------
@st.cache_resource
def initialize_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.warning(f"Groq client init failed: {e}")
        return None

groq_client = initialize_groq_client()

def call_groq_chat(client, messages):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            **MODEL_SETTINGS
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"

# -------------------------
# Local fallback AI generators
# -------------------------
def local_generate_single():
    items = [
        ("Grilled Chicken", 250, 30, 0, 10, ["B6","B12","Iron"]),
        ("Brown Rice (1 cup)", 210, 4, 45, 2, ["B1","Magnesium"]),
        ("Mixed Salad", 40, 2, 6, 1, ["Vitamin C","Vitamin K"]),
        ("Paneer Curry", 300, 18, 8, 20, ["Calcium","B12"]),
        ("Oatmeal with Banana", 220, 6, 40, 5, ["B6","Magnesium"])
    ]
    choice = random.sample(items, 2)
    items_text = ", ".join(c[0] for c in choice)
    calories = sum(c[1] for c in choice)
    protein = sum(c[2] for c in choice)
    carbs = sum(c[3] for c in choice)
    fat = sum(c[4] for c in choice)
    vitamins = sorted({v for c in choice for v in c[5]})
    lines = []
    lines.append(f"Items: {items_text}")
    lines.append(f"Calories: {calories} kcal")
    lines.append(f"Protein: {protein}g | Carbs: {carbs}g | Fat: {fat}g")
    lines.append(f"Vitamins/Minerals: {', '.join(vitamins)}")
    lines.append("Recommendation: Good balance — add vegetables for fiber.")
    return "\n".join(lines)

def local_generate_plan(goal, meal_type, gender, activity, calories):
    templates = {
        "Breakfast": ["Oatmeal with nuts", "Greek yogurt & fruit", "Veg omelette"],
        "Lunch": ["Grilled chicken + quinoa", "Paneer curry + roti", "Mixed grain bowl"],
        "Dinner": ["Grilled fish + veg", "Tofu stir-fry + brown rice", "Lentil soup + salad"],
        "Snack": ["Fruit bowl", "Roasted chickpeas", "Protein shake"]
    }
    chosen = random.sample(templates.get(meal_type, templates["Lunch"]), 2)
    approx = int((calories or 2000)/4)
    text = f"Meal: {meal_type}\nItems: {', '.join(chosen)}\nApprox calories for this meal: ~{approx} kcal\nNote: Adjust portion for goal {goal}."
    return text

# -------------------------
# Nutrition helpers
# -------------------------
def extract_calories(text: str) -> Optional[int]:
    m = re.search(r"(\d{2,5})\s*kcal", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def calculate_daily_calories(age: int, gender: str, activity: str, weight: Optional[float]=None, height: Optional[float]=None) -> int:
    if weight is None or weight <= 0:
        weight = 70 if gender == "Male" else 60
    if height is None or height <= 0:
        height = 170 if gender == "Male" else 160
    if gender == "Male":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    factors = {"Sedentary":1.2, "Moderate":1.55, "Active":1.75}
    return int(bmr * factors.get(activity, 1.55))

# -------------------------
# Barcode helpers (pyzbar)
# -------------------------
def decode_barcode_from_pil(pil_img: Image.Image) -> Optional[str]:
    if not PYZBAR_AVAILABLE:
        st.warning("pyzbar/opencv not available — install pyzbar and opencv-python for barcode decoding.")
        return None
    try:
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        decoded = zbar_decode(gray)
        if not decoded:
            decoded = zbar_decode(arr)
        if decoded:
            return decoded[0].data.decode("utf-8")
        return None
    except Exception as e:
        st.error(f"Barcode decode error: {e}")
        return None

def lookup_openfoodfacts(barcode: str) -> Optional[dict]:
    try:
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != 1:
            return None
        return data.get("product")
    except Exception as e:
        st.error(f"OpenFoodFacts request failed: {e}")
        return None

def pretty_nutrients(product: dict) -> dict:
    nutr = product.get("nutriments", {}) or {}
    fields = {
        "energy-kcal_100g":"Calories (kcal/100g)",
        "proteins_100g":"Protein (g/100g)",
        "fat_100g":"Fat (g/100g)",
        "carbohydrates_100g":"Carbs (g/100g)",
        "sugars_100g":"Sugars (g/100g)",
        "fiber_100g":"Fiber (g/100g)",
        "salt_100g":"Salt (g/100g)"
    }
    out = {}
    for k,label in fields.items():
        if k in nutr:
            out[label] = nutr.get(k)
    return out

# -------------------------
# Food Fact helpers
# -------------------------
def get_food_fact():
    # lightweight facts fallback to uselessfacts; can be swapped with API Ninjas if key provided
    try:
        response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en", timeout=6)
        if response.status_code == 200:
            data = response.json()
            return data.get("text", "Eat healthy — tip: include more vegetables.")
        else:
            return "Could not fetch fact right now. Try again."
    except Exception:
        return "Error fetching fact. Please check your connection."

def get_food_image():
    try:
        response = requests.get("https://foodish-api.herokuapp.com/api/", timeout=6)
        if response.status_code == 200:
            data = response.json()
            return data.get("image")
        else:
            return None
    except Exception:
        return None

# -------------------------
# AI Assistant KB (TF-IDF)
# -------------------------
_NUTRITION_KB = [
    {"q":"how much protein do i need daily","a":"Protein needs depend on body weight and activity. A general guide: sedentary: 0.8 g/kg, moderately active: 1.2–1.6 g/kg, muscle gain: 1.6–2.2 g/kg. Provide your weight for a precise number."},
    {"q":"suggest a 500 calorie breakfast","a":"Example 500 kcal breakfast: 1 cup cooked oats with 1 tbsp peanut butter and 1 medium banana (~500 kcal). For higher protein, add a scoop of whey or Greek yogurt."},
    {"q":"what foods are high in vitamin c","a":"High vitamin C foods: oranges, guava, strawberries, kiwi, bell peppers, broccoli."},
    {"q":"how to reduce sugar intake","a":"Reduce sugary drinks and packaged snacks, replace with whole fruits, check labels for added sugars."},
    {"q":"how to increase iron in diet","a":"Iron-rich foods: red meat, poultry, lentils, spinach, fortified cereals. Pair with vitamin C to improve absorption."},
    {"q":"what are macronutrients","a":"Macronutrients are protein, carbohydrates, and fats. Protein helps repair tissue, carbs provide energy, fats support hormones and cell function."},
]

def _build_kb_vectorizer():
    corpus = [item["q"] for item in _NUTRITION_KB]
    vect = TfidfVectorizer(ngram_range=(1,2)).fit(corpus)
    mat = vect.transform(corpus)
    return vect, mat

_VECT, _KB_MAT = _build_kb_vectorizer()

def _retrieve_kb_answer(query, threshold=0.35):
    vec_q = _VECT.transform([query])
    sims = cosine_similarity(vec_q, _KB_MAT).flatten()
    idx = sims.argmax()
    score = sims[idx]
    if score >= threshold:
        return _NUTRITION_KB[idx]["a"], float(score)
    return None, float(score)

def _extract_weight_from_text(text):
    m = re.search(r"(\d{2,3})\s*(kg|kilograms)?", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def _is_protein_question(text):
    return bool(re.search(r"\bprotein\b|\bhow much protein\b|\bneed protein\b", text, re.I))

def _is_calorie_question(text):
    return bool(re.search(r"\bcalorie\b|\bcalories\b|\bdaily calorie\b|\bcalorie need\b", text, re.I))

def _is_500cal_breakfast(text):
    return bool(re.search(r"500\s*calorie.*breakfast|suggest.*500.*breakfast", text, re.I))

def calc_protein_need(weight_kg, goal="Moderate"):
    if weight_kg is None:
        return None
    if goal.lower().startswith("sed"):
        factor = 0.8
    elif goal.lower().startswith("mus"):
        factor = 1.8
    else:
        factor = 1.3
    grams = weight_kg * factor
    return round(grams, 1), factor

def assistant_answer(query: str, user_info: dict = None):
    q = query.strip().lower()
    # protein intent
    if _is_protein_question(q):
        w = _extract_weight_from_text(q) or (user_info.get("weight") if user_info else None)
        goal = "Moderate"
        if "muscle" in q or (user_info and user_info.get("goal","").lower().find("muscle")>=0):
            goal = "Muscle"
        elif "lose" in q or "weight loss" in q:
            goal = "Sedentary"
        if w:
            grams, factor = calc_protein_need(w, goal)
            txt = f"Approx protein need: **{grams} g/day** (≈ {factor} g/kg × {w} kg) for goal *{goal}*."
            return txt, {"source":"calculation","protein_g":grams}
        else:
            return ("To calculate protein needs I need your weight (kg). Provide it like: 'I weigh 70 kg' or fill your profile."), {"source":"clarify"}

    # calorie intent
    if _is_calorie_question(q):
        if user_info and all(k in user_info for k in ("age","gender","activity")):
            cal = calculate_daily_calories(user_info["age"], user_info["gender"], user_info["activity"], user_info.get("weight"), user_info.get("height"))
            return f"Estimated daily calories (Mifflin–St Jeor): **{cal} kcal/day** for your profile.", {"source":"calculation","calories":cal}
        else:
            return "I can calculate your daily calorie needs — please provide age, gender, activity, and (optional) weight/height in your profile.", {"source":"clarify"}

    # 500 cal breakfast
    if _is_500cal_breakfast(q):
        ans = "Example ~500 kcal breakfast: **1 cup cooked oats (150g)** + **1 tbsp peanut butter** + **1 medium banana** (~500 kcal). For extra protein add Greek yogurt."
        return ans, {"source":"kb_template"}

    # kb retrieval
    kb_ans, score = _retrieve_kb_answer(query)
    if kb_ans:
        return kb_ans, {"source":"kb","score":score}

    return ("I couldn't find a direct answer — try rephrasing or provide profile details for personalized calculations."), {"source":"fallback"}

# -------------------------
# UI: Authentication: Login / Signup / Profile Setup
# -------------------------
def show_login_signup():
    
    st.title("Welcome to Smart Nutrition Analyzer 🥗")
    st.write("Login to continue or create a new account.")
    

    # Tabs for Login & Sign Up
    tabs = st.tabs(["Login", "Sign up"])
    show_forgot = st.session_state.get("show_forgot", False)
    

    # ---------- LOGIN ----------
    with tabs[0]:
        if not show_forgot:
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            col1, col2 = st.columns([3, 2])
            with col1:
                if st.button("Login", use_container_width=True):
                    if not email or not password:
                        st.error("Enter email and password.")
                    else:
                        users = load_users()
                        user = users.get(email)
                        if not user:
                            st.error("User not found. Please sign up.")
                        elif not check_password(password, user.get("password", "")):
                            st.error("Incorrect password.")
                        else:
                            st.success("Logged in successfully.")
                            st.session_state["logged_in"] = True
                            st.session_state["user_email"] = email
                            st.session_state["user_profile"] = user.get("profile", {})
                            safe_rerun()
            with col2:
                if st.button("Forgot Password?", use_container_width=True):
                    st.session_state["show_forgot"] = True
                    st.rerun()

        else:
            # ---------- FORGOT PASSWORD ----------
            st.subheader("🔒 Reset Password")
            email = st.text_input("Registered Email", key="forgot_email")
            new_pass = st.text_input("New Password", type="password", key="forgot_newpass")
            confirm_pass = st.text_input("Confirm New Password", type="password", key="forgot_confirmpass")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset Password", use_container_width=True):
                    if not email or not new_pass or not confirm_pass:
                        st.warning("Please fill all fields.")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        users = load_users()
                        if email not in users:
                            st.error("Email not registered. Please sign up.")
                        else:
                            users[email]["password"] = hash_password(new_pass)
                            save_users(users)
                            st.success("Password reset successfully! Please login again.")
                            st.session_state["show_forgot"] = False
                            st.rerun()
            with col2:
                if st.button("Back to Login", use_container_width=True):
                    st.session_state["show_forgot"] = False
                    st.rerun()

    # ---------- SIGNUP ----------
    with tabs[1]:
        st.subheader("Create a new account")
        new_email = st.text_input("Email", key="signup_email")
        new_pass = st.text_input("Password", type="password", key="signup_password")
        new_pass2 = st.text_input("Confirm Password", type="password", key="signup_password2")

        if st.button("Create Account"):
            if not new_email or not new_pass or not new_pass2:
                st.error("Please fill all fields.")
            elif new_pass != new_pass2:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_email in users:
                    st.error("Email already registered. Please login.")
                else:
                    hashed = hash_password(new_pass)
                    users[new_email] = {"password": hashed, "profile": {}}
                    save_users(users)
                    st.success("Account created! Please complete your profile.")
                    st.session_state["temp_signup_email"] = new_email
                    safe_rerun()


def show_profile_setup():
    st.header("Complete your profile")
    email = st.session_state.get("temp_signup_email")
    if not email:
        st.info("Start signup again.")
        if st.button("Back to Signup"):
            safe_rerun()
        return

    st.write("Tell us about yourself. This helps personalize recommendations.")
    name = st.text_input("Full name", key="profile_name")
    age = st.number_input("Age", min_value=10, max_value=100, value=25, key="profile_age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="profile_gender")
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5, key="profile_weight")
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5, key="profile_height")
    activity = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"], index=1, key="profile_activity")

    if st.button("Save Profile"):
        users = load_users()
        if email not in users:
            st.error("Signup session expired. Please sign up again.")
            return
        users[email]["profile"] = {
            "name": name,
            "age": int(age),
            "gender": gender,
            "weight": float(weight),
            "height": float(height),
            "activity": activity
        }
        save_users(users)
        st.success("Profile saved. You are now logged in.")
        st.session_state["logged_in"] = True
        st.session_state["user_email"] = email
        st.session_state["user_profile"] = users[email]["profile"]
        if "temp_signup_email" in st.session_state:
            st.session_state.pop("temp_signup_email")
        safe_rerun()




# -------------------------
# UI: Header & Sidebar (for logged in)
# -------------------------
def render_header_and_sidebar():
    logo_b64 = get_logo_b64()
    col1, col2 = st.columns([1, 6])
    with col1:
        if logo_b64:
            st.image(base64.b64decode(logo_b64), width=64)
    with col2:
        st.markdown("<h2 style='margin:0; color:#2e7d32'>Smart Nutrition Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<div style='color:#FF6347; margin-top:-8px'>AI-Powered Nutrition Insights</div>", unsafe_allow_html=True)
    st.markdown("---")

    # sidebar menu
    with st.sidebar:
        st.write(f"👤 **{st.session_state.get('user_profile', {}).get('name', st.session_state.get('user_email'))}**")
        st.markdown("---")
        selected = option_menu(
            "Menu",
            ["Home","Analyze Meal","Compare Meal","Plan Meal","Count Calorie","Scan Barcode","AI Assistant","Food Fact","Report History","Profile","Logout"],
            icons=["house","card-text","arrow-left-right","egg-fried","calculator","upc-scan","chat","lightbulb","clock-history","person","box-arrow-right"],
            menu_icon="menu-button",
            default_index=0,
            orientation="vertical",
        )
    return selected

# -------------------------
# MAIN APP PAGES (require logged in)
# -------------------------
def run_main_app(selected):
    # ---------- Analyze Meal ----------
    if selected == "Analyze Meal":
        st.header("🍽️ Analyze Meal")
        st.write("Upload a clear image of your meal. The app will analyze and give a short nutrition summary.")
        upload = st.file_uploader("Upload Meal Image", type=ALLOWED_FILE_TYPES, key="analyze_upload")
        if upload:
            try:
                pil = Image.open(upload)
                st.image(pil, caption="Uploaded Image", use_container_width=True)
                if st.button("Analyze Meal 🍎", use_container_width=True):
                    b64, fmt = pil_image_to_b64_and_fmt(pil)
                    if groq_client:
                        messages = [{"role":"user","content":[{"type":"text","text":prompt_for_single()},{"type":"image_url","image_url":{"url":f"data:image/{fmt.lower()};base64,{b64}"}}]}]
                        analysis = call_groq_chat(groq_client, messages)
                    else:
                        analysis = local_generate_single()
                    st.session_state["last_analysis"] = analysis
                    entry = {
                        "id": datetime.utcnow().isoformat(),
                        "type": "Single Analysis",
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": analysis,
                        "meals": [],
                        "calories": extract_calories(analysis),
                    }
                    add_history(entry)
                    st.success("Analysis Complete — saved to history.")
                    st.info(analysis)
                    pdf_buf = generate_pdf_from_entry("Meal Analysis", analysis)
                    st.download_button("📄 Download Report", data=pdf_buf, file_name="meal_analysis.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to open image: {e}")

    # ---------- Plan Meal ----------
    elif selected == "Plan Meal":
        st.header("🥗 Plan Meal (Goal-based, one meal)")
        st.write("Select goal and meal type — AI will generate a suggested meal for that single meal.")
        goal = st.selectbox("Goal", ["Weight Loss","Muscle Gain","Balanced Diet"])
        meal_type = st.selectbox("Meal Type", ["Breakfast","Lunch","Dinner","Snack"])
        gender = st.selectbox("Gender (optional)", ["Male","Female","Other"])
        activity = st.selectbox("Activity Level", ["Sedentary","Moderate","Active"])
        cal_input = st.number_input("Daily calorie target (optional)", min_value=800, max_value=4000, value=800, step=50)
        if st.button("Generate Plan 🍽️", use_container_width=True):
            if groq_client:
                messages = [{"role":"user","content":prompt_for_plan(goal, meal_type, gender, activity, cal_input or None)}]
                plan_text = call_groq_chat(groq_client, messages)
            else:
                plan_text = local_generate_plan(goal, meal_type, gender, activity, cal_input or None)
            st.session_state["last_plan"] = plan_text
            st.success("Meal Plan Generated — saved to history.")
            entry = {
                "id": datetime.utcnow().isoformat(),
                "type": "Meal Plan",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "text": plan_text,
                "meals": [meal_type],
                "calories": None
            }
            add_history(entry)
            st.info(plan_text)
            pdf_buf = generate_pdf_from_entry(f"{goal} - {meal_type} Plan", plan_text)
            st.download_button("📄 Download Meal Plan", data=pdf_buf, file_name=f"{goal}_{meal_type}_plan.pdf", mime="application/pdf")

    # ---------- Compare Meal ----------
    elif selected == "Compare Meal":
        st.header("⚖️ Compare Meals")
        st.write("Analyze Meal A first (Analyze Meal tab). Then come here to upload Meal B to compare.")
        if "last_analysis" not in st.session_state:
            st.warning("No analyzed Meal A found. Please analyze Meal A first.")
        else:
            st.markdown("**Stored Meal A (summary)**")
            st.code(st.session_state["last_analysis"], language="text")
            upload_b = st.file_uploader("Upload Meal B Image", type=ALLOWED_FILE_TYPES, key="compare_upload")
            if upload_b:
                pil_b = Image.open(upload_b)
                st.image(pil_b, caption="Meal B", use_container_width=True)
                if st.button("Compare Meals ⚖️", use_container_width=True):
                    b64b, fmtt = pil_image_to_b64_and_fmt(pil_b)
                    if groq_client:
                        messages = [{"role":"user","content":[{"type":"text","text":prompt_for_compare(st.session_state["last_analysis"])},{"type":"image_url","image_url":{"url":f"data:image/{fmtt.lower()};base64,{b64b}"}}]}]
                        comparison = call_groq_chat(groq_client, messages)
                    else:
                        comparison = "🥗 Meal Comparison Summary\n- Better for protein: Similar\n- Better for calories: Meal B slightly higher\n- Vitamins: Meal A richer in B vitamins\nRecommendation: Choose more vegetables."
                    st.session_state.pop("last_analysis", None)
                    st.session_state["last_comparison"] = comparison
                    entry = {
                        "id": datetime.utcnow().isoformat(),
                        "type": "Meal Comparison",
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": comparison,
                        "meals": [],
                        "calories": None
                    }
                    add_history(entry)
                    st.success("Comparison done — saved to history.")
                    st.info(comparison)
                    pdf_buf = generate_pdf_from_entry("Meal Comparison", comparison)
                    st.download_button("📄 Download Comparison", data=pdf_buf, file_name="meal_comparison.pdf", mime="application/pdf")

    # ---------- Count Calorie ----------
    elif selected == "Count Calorie":
        st.header("🏃‍♂️ Daily Calorie Recommendation")
        age = st.number_input("Age", min_value=10, max_value=100, value=int(st.session_state.get("user_profile", {}).get("age", 25)))
        gender = st.selectbox("Gender", ["Male","Female"], index=0)
        activity = st.selectbox("Activity Level", ["Sedentary","Moderate","Active"], index=1)
        weight = st.number_input("Weight (kg) — optional", min_value=0.0, max_value=300.0, value=float(st.session_state.get("user_profile", {}).get("weight", 0.0)))
        height = st.number_input("Height (cm) — optional", min_value=0.0, max_value=250.0, value=float(st.session_state.get("user_profile", {}).get("height", 0.0)))
        if st.button("Calculate Daily Calories", use_container_width=True):
            w = weight if weight > 0 else None
            h = height if height > 0 else None
            daily = calculate_daily_calories(int(age), gender, activity, w, h)
            st.session_state["user_daily_calories"] = daily
            st.success(f"Estimated daily calorie need: {daily} kcal (approx.)")

    # ---------- Report History ----------
    elif selected == "Report History":
        st.header("📜 Report History")
        hist = load_history()
        if not hist:
            st.info("No saved reports yet. Generate analysis or plans to populate history.")
        else:
            for idx, entry in enumerate(hist):
                st.markdown(
                    f"<div style='padding:10px; border-radius:8px; background:linear-gradient(90deg,#fff,#fbfbfb); margin-bottom:10px;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<div><b>{entry.get('type','Report')}</b> &nbsp; • &nbsp; <span style='color:#666'>{entry.get('timestamp','')}</span></div>"
                    f"</div>"
                    f"<div style='margin-top:8px; white-space:pre-wrap; font-family:inherit; font-size:14px'>{entry.get('text','')}</div>"
                    f"</div>", unsafe_allow_html=True
                )
                if st.button(f"📥 Download PDF — {idx}", key=f"hist_pdf_{idx}"):
                    pdf_buf = generate_pdf_from_entry(entry.get("type","Report"), entry.get("text",""))
                    st.download_button("Click to save PDF", data=pdf_buf, file_name=f"{entry.get('type','report')}_{idx}.pdf", mime="application/pdf")
            if st.button("🗑️ Clear All History", use_container_width=True):
                clear_history_file()
                safe_rerun()

    # ---------- Scan Barcode ----------
    elif selected == "Scan Barcode":
        st.header("📦 Scan Food Barcode")
        st.write("Upload a photo of the barcode (EAN/UPC). The app will decode and fetch product data from OpenFoodFacts (free).")
        uploaded_bar = st.file_uploader("Upload barcode image (jpg/png)", type=ALLOWED_FILE_TYPES, key="barcode_upload")
        if uploaded_bar:
            try:
                pil_img = Image.open(uploaded_bar).convert("RGB")
                st.image(pil_img, caption="Uploaded Barcode Image", use_container_width=False)
                if st.button("Decode & Lookup"):
                    with st.spinner("Decoding barcode and looking up product..."):
                        code = decode_barcode_from_pil(pil_img)
                        if not code:
                            st.warning("No barcode detected. Try cropping the barcode area or upload a clearer photo.")
                        else:
                            st.success(f"Detected barcode: {code}")
                            product = lookup_openfoodfacts(code)
                            if not product:
                                st.info("Product not found on OpenFoodFacts.")
                            else:
                                name = product.get("product_name") or product.get("generic_name") or "Unknown product"
                                brand = product.get("brands", "")
                                nutr = pretty_nutrients(product)
                                st.subheader(name)
                                if brand:
                                    st.write(f"**Brand:** {brand}")
                                if product.get("image_small_url"):
                                    st.image(product.get("image_small_url"), width=150)
                                st.markdown("**Nutrition (per 100g):**")
                                for k,v in nutr.items():
                                    st.write(f"- {k}: {v}")
                                text = f"{name} ({brand})\n" + "\n".join([f"{k}: {v}" for k,v in nutr.items()])
                                entry = {
                                    "id": datetime.utcnow().isoformat(),
                                    "type": "Scanned Product",
                                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                    "text": text,
                                    "barcode": code,
                                    "meals": [],
                                    "calories": nutr.get("Calories (kcal/100g)")
                                }
                                add_history(entry)
                                st.success("Product added to history.")
                                pdf_buf = generate_pdf_from_entry(name, text)
                                st.download_button("📄 Download Product Report", data=pdf_buf, file_name=f"{name}_product_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to process barcode image: {e}")

    # ---------- AI Assistant ----------
    elif selected == "AI Assistant":
        st.header("💬 AI Chatbot")
        if "ai_chat" not in st.session_state:
            st.session_state["ai_chat"] = []
        for role, text in st.session_state["ai_chat"]:
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")
        user_q = st.text_input("Ask the nutrition assistant...", key="ai_input")
        if st.button("Send", key="ai_send") and user_q.strip():
            st.session_state.ai_chat.append(("user", user_q))
            user_info = st.session_state.get("user_profile", {})
            ans, meta = assistant_answer(user_q, user_info=user_info)
            st.session_state.ai_chat.append(("assistant", ans))
            add_history({
                "id": datetime.utcnow().isoformat(),
                "type": "AI Assistant Chat",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "text": f"Q: {user_q}\nA: {ans}"
            })
            safe_rerun()

    # ---------- Food Fact ----------
    elif selected == "Food Fact":
        st.header("🧠 Fun Food & Nutrition Facts")
        def safe_get_food_image():
            img = get_food_image()
            if not img or not isinstance(img, str) or not img.startswith("http"):
                return "https://cdn-icons-png.flaticon.com/512/2921/2921822.png"
            return img

        def get_filtered_fact():
            fact = get_food_fact()
            keywords = ["food", "diet", "nutrition", "vitamin", "fruit", "vegetable", "protein", "calorie", "health"]
            for _ in range(6):
                if any(k in fact.lower() for k in keywords):
                    return fact
                fact = get_food_fact()
            return fact

        if "food_fact" not in st.session_state:
            st.session_state["food_fact"] = get_filtered_fact()
        if "food_image" not in st.session_state:
            st.session_state["food_image"] = safe_get_food_image()

        st.markdown(
            f"""
            <div style="text-align:center; padding:18px; background-color:#f9fff9; 
                 border-radius:12px; box-shadow:0 4px 8px rgba(0,0,0,0.06);">
                <h3 style="color:#2E8B57; margin:6px 0;">🍀 Fun Nutrition Fact</h3>
                <p style="font-size:16px; color:#333; font-weight:500; margin:0 10px;">
                    {st.session_state['food_fact']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.get("food_image"):
            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-top:12px;">
                    <img src="{st.session_state['food_image']}" width="240"
                         style="border-radius:12px; box-shadow:0px 4px 12px rgba(0,0,0,0.08);" alt="Food Image" />
                </div>
                <p style="text-align:center; color:gray; font-style:italic; margin-top:6px;">🥗 Food Inspiration</p>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr style='width:50%; margin:auto; border:1px solid #eee;'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.3, 0.8, 1])
        with col2:
            st.markdown(
                """
                <style>
                div.stButton > button:first-child {
                    background-color:#2E8B57;
                    color:white;
                    font-size:15px;
                    font-weight:600;
                    border-radius:10px;
                    padding:10px 26px;
                    box-shadow: 0px 6px 14px rgba(46,139,87,0.18);
                    transition: transform .12s ease;
                }
                div.stButton > button:first-child:hover {
                    transform: translateY(-3px);
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button("✨ New Fact"):
                st.session_state['food_fact'] = get_filtered_fact()
                # force new image (append timestamp query if same)
                new_img = safe_get_food_image()
                if new_img == st.session_state.get("food_image"):
                    new_img = new_img + f"?v={int(time.time())}"
                st.session_state['food_image'] = new_img
                safe_rerun()

    # ---------- Profile ----------
    elif selected == "Profile":
        st.header("👤 Profile")
        profile = st.session_state.get("user_profile", {})
        st.write("Manage your profile and preferences.")
        st.text_input("Full name", value=profile.get("name",""), key="profile_name_edit")
        st.number_input("Age", min_value=10, max_value=100, value=int(profile.get("age",25)), key="profile_age_edit")
        st.selectbox("Gender", ["Male","Female","Other"], index=0, key="profile_gender_edit")
        st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=float(profile.get("weight",70.0)), key="profile_weight_edit")
        st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(profile.get("height",170.0)), key="profile_height_edit")
        st.selectbox("Activity Level", ["Sedentary","Moderate","Active"], index=1, key="profile_activity_edit")
        if st.button("Save Profile Changes"):
            users = load_users()
            email = st.session_state.get("user_email")
            if email and email in users:
                users[email]["profile"] = {
                    "name": st.session_state.get("profile_name_edit"),
                    "age": int(st.session_state.get("profile_age_edit")),
                    "gender": st.session_state.get("profile_gender_edit"),
                    "weight": float(st.session_state.get("profile_weight_edit")),
                    "height": float(st.session_state.get("profile_height_edit")),
                    "activity": st.session_state.get("profile_activity_edit")
                }
                save_users(users)
                st.session_state["user_profile"] = users[email]["profile"]
                st.success("Profile updated.")
            else:
                st.error("Could not update profile.")
                
    # ---------- Logout ----------
    elif selected == "Logout":
        st.header("Logout")
        if st.button("Confirm Logout"):
            # clear session login-related keys
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = None
            st.session_state["user_profile"] = {}
            st.success("You are logged out.")
            safe_rerun()

    # ---------- Home ----------
    elif selected == "Home":
       st.markdown(
        """
        <div style="text-align:center;">
            <h1 style='color:#2E8B57;margin-bottom:6px;'>🥗 Smart Nutrition Analyzer</h1>
            <p style='color:#444;margin-top:0px;font-size:17px;'>Your personal AI-powered nutrition assistant</p>
        </div>
        """,
        unsafe_allow_html=True
        )

    # --- Optional Lottie Animation ---
    if LOTTIE_AVAILABLE:
        def load_lottieurl(url):
            try:
                r = requests.get(url, timeout=6)
                if r.status_code == 200:
                    return r.json()
            except:
                return None
        lottie_url = "https://assets7.lottiefiles.com/packages/lf20_eyetg5os.json"
        lottie_json = load_lottieurl(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=280, key="home")

    # --- Section Header ---
    st.markdown("### ✨ Explore Our Key Features")

    # --- Add Custom CSS for Feature Cards ---
    st.markdown("""
        <style>
        .feature-card {
            padding: 18px;
            border-radius: 15px;
            background: linear-gradient(145deg, #f0fff0, #ffffff);
            box-shadow: 0 4px 15px rgba(46,139,87,0.2);
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(46,139,87,0.15);
        }
        .feature-card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(46,139,87,0.3);
        }
        .feature-icon {
            font-size: 32px;
        }
        .feature-text {
            margin-top: 8px;
            font-size: 16px;
            color: #333;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Three Feature Cards ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🍛</div>
                <div class="feature-text">Analyze your meal from a photo</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚖️</div>
                <div class="feature-text">Compare meals & choose healthier</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-text">AI meal planner & assistant</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:gray;'>Made with ....🥗</p>", unsafe_allow_html=True)




# -------------------------
# PROMPT utils (re-used)
# -------------------------
def prompt_for_single():
    return textwrap.dedent("""
        As an expert nutritionist with advanced image analysis capabilities, Provide a short, precise analysis for the food image.
        Output (concise):
        - Items: comma separated
        - Calories: N kcal
        - Macronutrients: Protein Ng | Carbs Ng | Fat Ng
        - Vitamins/Minerals: list only those present
        - Recommendation: one short sentence
        Keep it minimal and readable.
    """)

def prompt_for_compare(meal1_text: str):
    return textwrap.dedent(f"""
        You are an expert nutritionist. Compare Meal A (below) with the new image Meal B.
        Meal A summary:
        {meal1_text}

        Output a short comparison:
        - Better for protein: Meal A / Meal B / Similar
        - Better for calories: Meal A / Meal B / Similar
        - Vitamins/minerals: which meal has more
        - Recommendation: one short sentence (which to prefer and why)
        Keep it concise.
    """)

def prompt_for_plan(goal: str, meal_type: str, gender: str, activity: str, calories: Optional[int]):
    cal_text = f"{calories} kcal" if calories else "approx daily calories"
    return textwrap.dedent(f"""
        You are an AI nutrition planner. Create a one-meal plan given:
        Goal: {goal}
        Meal Type: {meal_type}
        Gender: {gender}
        Activity Level: {activity}
        Daily Calories: {cal_text}

        Provide:
        - Meal title
        - 2-4 food items with portion sizes
        - Approx calories for the meal
        - Short nutrients note (protein/carbs/fat)
        - One-line tip
        Keep response concise and user-friendly.
    """)

# -------------------------
# APP START: Authentication gating
# -------------------------
def main():
    st.markdown("<style>footer {visibility: hidden;} </style>", unsafe_allow_html=True)
    # if user not logged in -> show login/signup or profile setup
    if not st.session_state.get("logged_in", False):
        # check if we are in "complete profile after signup" state
        if st.session_state.get("temp_signup_email"):
            show_profile_setup()
        else:
            show_login_signup()
        # end early until login
        return

    # if logged in, render header + sidebar + main pages
    selected = render_header_and_sidebar()
    run_main_app(selected)

    # footer
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#888;'>© Smart Nutrition Analyzer</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
