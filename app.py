from flask import Flask, jsonify
import joblib
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import yfinance as yf
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)

# ==========================================
# 1. DATABASE & SECURITY CONFIGURATION
# ==========================================
# This reads the database URL from Render. If testing locally, it uses a local file.
db_url = os.environ.get('DATABASE_URL', 'sqlite:///kasu_wealth.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1) # SQLAlchemy requires 'postgresql://'

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET', 'super-secret-kasu-key-change-this-later')

db = SQLAlchemy(app)
jwt = JWTManager(app)

# ==========================================
# 2. DATABASE TABLES (MODELS)
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    # This links the user to their purchases
    purchases = db.relationship('Purchase', backref='owner', lazy=True)

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_name = db.Column(db.String(100), nullable=False)  # e.g., "Reliance", "Gold"
    buy_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(50), nullable=False)

# Initialize the database tables
with app.app_context():
    db.create_all()

# ==========================================
# 3. AUTHENTICATION ROUTES
# ==========================================

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email is already registered"}), 400

    # Scramble the password so hackers can't read it
    hashed_pw = generate_password_hash(password)
    
    new_user = User(name=name, email=email, password_hash=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True, "message": "Account created successfully!"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    
    # Check if user exists AND password matches the scrambled hash
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"success": False, "message": "Invalid email or password"}), 401

    # Create the secure digital key (JWT)
    access_token = create_access_token(identity=str(user.id))
    
    return jsonify({
        "success": True, 
        "token": access_token, 
        "name": user.name
    }), 200

# Example of a PROTECTED route (requires the JWT token to access)
@app.route('/api/my_portfolio', methods=['GET'])
@jwt_required()
def my_portfolio():
    # Find out who is making the request using their token
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    # Fetch all purchases for this specific user
    user_purchases = Purchase.query.filter_by(user_id=user.id).all()
    
    portfolio_data = [{"asset": p.asset_name, "price": p.buy_price, "qty": p.quantity} for p in user_purchases]
    
    return jsonify({"success": True, "owner": user.name, "portfolio": portfolio_data}), 200

# ... (Keep all your existing AI routes down here) ...

app = Flask(__name__)

# --- HEALTH CHECK (Answers Render instantly!) ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Kasu Wealth AI Backend is Live & Healthy!"}), 200

# ==========================================
# 1. AI MODELS & LAZY LOADING
# ==========================================
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Global variables to hold models
model = None
q_network = None

def load_models():
    """Loads models ONLY when requested, preventing server boot crashes!"""
    global model, q_network
    if model is None:
        try:
            model = joblib.load('mutual_fund_tree.pkl')
            print("✅ Decision Tree loaded!")
        except Exception as e:
            print(f"❌ Tree Error: {e}")
            
    if q_network is None:
        try:
            q_network = QNetwork()
            q_network.load_state_dict(torch.load('q_network.pth', weights_only=True))
            q_network.eval()
            print("✅ Q-Network loaded!")
        except Exception as e:
            print(f"❌ Q-Network Error: {e}")

# ==========================================
# 2. ASSET DATA
# ==========================================
funds = {
    "119383": "HDFC Top 100 Fund",
    "119911": "HDFC Flexi Cap Fund", 
    "119946": "HDFC Mid-Cap Opportunities Fund",
    "120577": "SBI Bluechip Fund"
}

GOLD_ETFS = ["SBI Gold ETF", "Nippon India ETF Gold BeES", "HDFC Gold ETF"]
SILVER_ETFS = ["Nippon India Silver ETF", "ICICI Prudential Silver ETF", "HDFC Silver ETF", "SBI Silver ETF", "Tata Silver ETF"]
RE_ETFS = ["ICICI Prudential Nifty Next 50", "Kotak Nifty Next 50", "HDFC Nifty Next 50"]

# ==========================================
# 3. PORTFOLIO ANALYZER ENDPOINT
# ==========================================
@app.route('/analyze_portfolio', methods=['GET'])
def analyze_portfolio():
    # ⚡ ONLY LOAD MODELS WHEN SOMEONE ASKS FOR AN ANALYSIS
    load_models()
    
    # If models failed to load, return safe fallback data so app doesn't crash
    if model is None or q_network is None:
         return jsonify({"success": True, "results": [{"fund_name": name, "fund_id": code, "status": "Continue"} for code, name in funds.items()]})

    tree_results = []
    pause_count = 0
    
    for fund_code, fund_name in funds.items():
        features = pd.DataFrame({
            model.feature_names_in_[0]: [0.97 if "HDFC" in fund_name else 1.02],
            model.feature_names_in_[1]: [0.02],
            model.feature_names_in_[2]: [0.001],
            model.feature_names_in_[3]: [0],
            model.feature_names_in_[4]: [5000],
            model.feature_names_in_[5]: [0.0],
            model.feature_names_in_[6]: [0.0]
        })
        
        prediction = model.predict(features)[0]
        decision = ("Stop" if prediction == 2 else "Pause" if prediction == 0 else "Continue")
        
        if decision in ["Pause", "Stop"]:
            pause_count += 1
            
        tree_results.append({"fund_code": fund_code, "fund_name": fund_name, "action": decision})

    total_funds = len(tree_results) or 1
    pause_ratio = pause_count / total_funds
    final_output = []
    
    if pause_ratio >= 0.5:
        state = np.array([pause_ratio, 0, 1 - pause_ratio, 0.018, 0.0234, 0.045, 0.0057, 0.145, 0.22], dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor).numpy()[0]
        
        best_idx = np.argmax(q_values)
        pools = [GOLD_ETFS, SILVER_ETFS, RE_ETFS, []]
        etf_pool = pools[best_idx].copy() if best_idx < 3 else []
        
        if etf_pool:
            random.shuffle(etf_pool)
            etf_index = 0
            for fund in tree_results:
                if fund['action'] in ['Pause', 'Stop']:
                    etf_name = etf_pool[etf_index % len(etf_pool)]
                    status = f"Pause, going to shift to real estate: {etf_name}" if best_idx == 2 else f"Pause, going to shift: {etf_name}"
                    etf_index += 1
                else:
                    status = fund['action']
                final_output.append({"fund_name": fund['fund_name'], "fund_id": fund['fund_code'], "status": status})
        else:
            final_output = [{"fund_name": f['fund_name'], "fund_id": f['fund_code'], "status": f['action']} for f in tree_results]
    else:
        final_output = [{"fund_name": f['fund_name'], "fund_id": f['fund_code'], "status": f['action']} for f in tree_results]
    
    return jsonify({"success": True, "results": final_output})

# ==========================================
# 4. LIVE REAL-WORLD MARKET DATA 
# ==========================================
@app.route('/api/live_market', methods=['GET'])
def live_market():
    try:
        gold_data = yf.Ticker("GC=F").history(period="7d")['Close'].tolist()
        silver_data = yf.Ticker("SI=F").history(period="7d")['Close'].tolist()

        if not gold_data or not silver_data:
            raise ValueError("Yahoo Finance empty")

        gold_inr = [round(p * 2.668, 2) for p in gold_data]
        silver_inr = [round(p * 2.668, 2) for p in silver_data]

        nifty_etf = yf.Ticker("NIFTYBEES.NS").history(period="1d")
        gold_etf = yf.Ticker("GOLDBEES.NS").history(period="1d")

        nifty_price = round(nifty_etf['Close'].iloc[-1], 2) if not nifty_etf.empty else 245.50
        gold_etf_price = round(gold_etf['Close'].iloc[-1], 2) if not gold_etf.empty else 54.20

        return jsonify({
            "success": True,
            "metals": {
                "gold": {"price": f"₹{gold_inr[-1]:,.2f}/gm", "chart": gold_inr},
                "silver": {"price": f"₹{silver_inr[-1]:,.2f}/gm", "chart": silver_inr}
            },
            "etfs": {"NIFTYBEES": f"₹{nifty_price}", "GOLDBEES": f"₹{gold_etf_price}"}
        })
    except Exception as e:
        # BULLETPROOF FALLBACK
        fallback_gold = [13500.0, 13550.0, 13480.0, 13600.0, 13620.0, 13650.0, 13668.46]
        fallback_silver = [230.0, 232.0, 231.5, 235.0, 234.0, 236.0, 237.34]
        return jsonify({
            "success": True, 
            "metals": {
                "gold": {"price": f"₹{fallback_gold[-1]:,.2f}/gm", "chart": fallback_gold},
                "silver": {"price": f"₹{fallback_silver[-1]:,.2f}/gm", "chart": fallback_silver}
            },
            "etfs": {"NIFTYBEES": "₹245.50", "GOLDBEES": "₹54.20"}
        })

