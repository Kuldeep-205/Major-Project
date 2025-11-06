# Major-Project
🧠 AI-Powered Nutrition Analyzer

AI-Powered Nutrition Analyzer is an intelligent, interactive Streamlit-based application that uses AI and computer vision to analyze food images, estimate calorie and nutrient values, and provide smart meal insights.
It combines nutrition awareness, AI-powered analysis, and user personalization — all in one beautifully designed web app. 🍽️✨

🌟 Key Features

✅ AI-Powered Meal Analysis
Upload a meal image, and the system identifies food items and provides detailed calorie & nutrient estimates.

✅ Barcode Scanner
Quickly fetch nutrition data by scanning packaged food barcodes.

✅ Smart Calorie Counter
Track your daily intake with automatic calorie logging and recommendations.

✅ AI Meal Planner
Get personalized meal suggestions based on your profile, activity level, and goals.

✅ Profile Management
Users can sign up, log in, and create profiles with height, weight, gender, and activity preferences for personalized analytics.

✅ Authentication System
Includes Login, Signup, and Forgot Password functionalities using secure password hashing and persistent storage.

✅ Modern UI with Theming
Features semi-transparent glass-style cards, red-accent buttons, and a full-screen background image on the welcome screen.

✅ Report Generation
Generate and download PDF nutrition reports summarizing your analyzed meals.

✅ Data Persistence
User profiles and credentials are securely stored in a local JSON database.

🧩 Project Structure
AI-Nutrition-Analyzer/
│
├── src/                 # Background and static images
├── test_img/            # Test food images for analysis
├── venv/                # Virtual environment
│
├── .env                 # Environment file (API keys if used)
├── .gitignore           # Files and folders ignored by Git
├── app.py               # Main Streamlit application file
├── LICENSE              # License file
├── README.md            # Project documentation (this file)
├── requirements.txt     # Required dependencies
└── users.json           # Local database for user authentication

⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/Kuldeep-205/Major-Project.git
cd Major-Project

2️⃣ Create a Virtual Environment
python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt



4️⃣ Add Environment Variables

Create a .env file in the project root directory and add your Groq API key:

GROQ_API_KEY= your_api_key_here ---(just edit this)
MODEL_TEMPERATURE=0.2
MODEL_MAX_TOKENS=400
MODEL_TOP_P=0.5

**Note :--
🌐 Step-by-Step: Create a GROQ API Key
1️⃣ Go to GROQ’s official site

👉 https://console.groq.com

2️⃣ Sign up or log in

You can use your Google account or email to create a free account.

3️⃣ Go to API Keys section

After logging in:

Click your profile icon (top-right) → API Keys

Click “+ New Key” or “Create Key”

4️⃣ Copy your API Key and pase in .env section



🚀 Usage

Run the Streamlit app:

streamlit run app.py


Then open in your browser:
👉 http://localhost:8501

Steps to Use:

Sign up or log in to access the main dashboard.

Upload a food image or scan a barcode to analyze meals.

View calorie and nutrition data.

Generate or download your meal report.

Plan your next meal using the AI Meal Planner.


📦 requirements.txt (for reference)
streamlit
opencv-python
pandas
numpy
scikit-learn
Pillow
requests
python-dotenv
bcrypt
fpdf
matplotlib
streamlit-lottie


