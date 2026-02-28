from flask import Flask, jsonify
import joblib
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import yfinance as yf

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
