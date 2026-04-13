from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler # 🚀 CRON JOB ENGINE
from datetime import datetime, timedelta # 🚀 TIME MANAGEMENT
import os
import random
import yfinance as yf
from mftool import Mftool
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app) # Allows your Flutter app to talk to Python securely

@app.route('/')
def home():
    return "🚀 Kasu Backend API is LIVE and running smoothly on Docker!"

# ==========================================
# 🚀 1. SECURE DATABASE CONFIGURATION
# ==========================================
database_url = os.environ.get('DATABASE_URL', 'sqlite:///local_database.db')
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-wealth-key-2026')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Initialize Mftool for Indian Mutual Funds
mf = Mftool()

# ==========================================
# 🚀 2. DATABASE MODELS (THE TABLES)
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    balance = db.Column(db.Float, default=0.0)

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_name = db.Column(db.String(100), nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(50), nullable=False)

# 🚀 AUTOMATED SIP TABLE
class SIP(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_name = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    next_due_date = db.Column(db.String(50), nullable=False)
    is_active = db.Column(db.Boolean, default=True)

# ==========================================
# 🚀 3. AUTHENTICATION (LOGIN & REGISTER)
# ==========================================
@app.route('/api/reset_db')
def reset_db():
    db.drop_all()   
    db.create_all() 
    return "Database successfully reset and synced with new architecture!"

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if User.query.filter_by(email=data['email']).first():
            return jsonify({"success": False, "message": "Email already registered."}), 400
            
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        new_user = User(full_name=data['full_name'], email=data['email'], password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        # Auto-create a wallet for the new user
        new_wallet = Wallet(user_id=new_user.id, balance=0.0)
        db.session.add(new_wallet)
        db.session.commit()

        return jsonify({"success": True, "message": "Account created successfully!"}), 201
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user = User.query.filter_by(email=data['email']).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, data['password']):
            access_token = create_access_token(identity=str(user.id))
            return jsonify({
                "success": True, 
                "token": access_token, 
                "user_name": user.full_name,
                "message": "Login successful!"
            }), 200
        else:
            return jsonify({"success": False, "message": "Invalid email or password."}), 401
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/user_profile', methods=['GET'])
@jwt_required()
def user_profile():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if user:
            return jsonify({"success": True, "user_name": user.full_name}), 200
        return jsonify({"success": False, "message": "User not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 🚀 4. WALLET & FINANCIAL TRANSACTIONS
# ==========================================
@app.route('/api/wallet', methods=['GET', 'POST'])
@jwt_required()
def manage_wallet():
    current_user_id = get_jwt_identity()
    wallet = Wallet.query.filter_by(user_id=current_user_id).first()
    
    if not wallet:
        wallet = Wallet(user_id=current_user_id, balance=0.0)
        db.session.add(wallet)
        db.session.commit()

    if request.method == 'GET':
        return jsonify({"success": True, "balance": wallet.balance}), 200
        
    if request.method == 'POST':
        try:
            data = request.get_json()
            amount = float(data.get('amount', 0.0))
            if amount <= 0:
                return jsonify({"success": False, "message": "Invalid amount."}), 400
                
            wallet.balance += amount
            db.session.commit()
            return jsonify({"success": True, "balance": wallet.balance, "message": f"Successfully added ₹{amount:,.2f} to wallet!"}), 200
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/buy_asset', methods=['POST'])
@jwt_required()
def buy_asset():
    try:
        user_id = get_jwt_identity()
        wallet = Wallet.query.filter_by(user_id=user_id).first()
        data = request.get_json()
        
        buy_price = float(data.get('buy_price'))
        quantity = float(data.get('quantity'))
        total_cost = buy_price * quantity

        if not wallet or wallet.balance < total_cost:
            bal = wallet.balance if wallet else 0.0
            return jsonify({"success": False, "message": f"Insufficient funds! Cost: ₹{total_cost:,.2f}, Balance: ₹{bal:,.2f}."}), 400

        wallet.balance -= total_cost
        new_purchase = Purchase(
            user_id=user_id, 
            asset_name=data.get('asset_name'), 
            buy_price=buy_price, 
            quantity=quantity, 
            date=data.get('date')
        )
        
        db.session.add(new_purchase)
        db.session.commit()
        return jsonify({"success": True, "message": f"Successfully purchased {quantity} units!"}), 201
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/sell_asset', methods=['POST'])
@jwt_required()
def sell_asset():
    try:
        user_id = get_jwt_identity()
        wallet = Wallet.query.filter_by(user_id=user_id).first()
        data = request.get_json()
        
        asset_name = data.get('asset_name')
        sell_price = float(data.get('sell_price'))
        sell_qty = float(data.get('quantity'))
        
        purchases = Purchase.query.filter_by(user_id=user_id, asset_name=asset_name).all()
        total_owned = sum(p.quantity for p in purchases)
        
        if total_owned < sell_qty:
            return jsonify({"success": False, "message": f"You only own {total_owned} units of {asset_name}!"}), 400
            
        qty_to_remove = sell_qty
        for p in purchases:
            if qty_to_remove <= 0: break
            if p.quantity <= qty_to_remove:
                qty_to_remove -= p.quantity
                db.session.delete(p)
            else:
                p.quantity -= qty_to_remove
                qty_to_remove = 0
                
        total_sale_value = sell_price * sell_qty
        wallet.balance += total_sale_value
        db.session.commit()
        
        return jsonify({"success": True, "message": f"Successfully sold {sell_qty} units for ₹{total_sale_value:,.2f}!"}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 🚀 5. PORTFOLIO & LEDGER DATA
# ==========================================
@app.route('/api/my_portfolio', methods=['GET'])
@jwt_required()
def my_portfolio():
    try:
        purchases = Purchase.query.filter_by(user_id=get_jwt_identity()).all()
        portfolio_list = [{"asset": p.asset_name, "price": p.buy_price, "qty": p.quantity, "date": p.date} for p in purchases]
        return jsonify({"success": True, "portfolio": portfolio_list}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/history', methods=['GET'])
@jwt_required()
def get_history():
    try:
        purchases = Purchase.query.filter_by(user_id=get_jwt_identity()).order_by(Purchase.id.desc()).all()
        history_list = [{
            "asset": p.asset_name.replace('[STOCK] ', '').replace('[GOLD] ', '').replace('[SILVER] ', '').replace('[MF] ', ''), 
            "type": "BUY/SELL", 
            "price": p.buy_price,
            "qty": p.quantity, 
            "total": p.buy_price * p.quantity, 
            "date": p.date
        } for p in purchases]
        
        return jsonify({"success": True, "history": history_list}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 🚀 6. AUTOMATED SIP ENGINE & CHARTS
# ==========================================
@app.route('/api/start_sip', methods=['POST'])
@jwt_required()
def start_sip():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        wallet = Wallet.query.filter_by(user_id=user_id).first()
        amount = float(data.get('amount'))
        asset_name = data.get('asset_name')
        
        if not wallet or wallet.balance < amount:
            return jsonify({"success": False, "message": "Insufficient funds for initial SIP deduction!"}), 400
            
        # 1. Deduct & Buy Initial Month Instantly
        wallet.balance -= amount
        new_purchase = Purchase(user_id=user_id, asset_name=asset_name, buy_price=amount, quantity=1.0, date=datetime.now().strftime('%Y-%m-%d'))
        db.session.add(new_purchase)
        
        # 2. Register Automated SIP for Future Months
        next_due = datetime.now() + timedelta(days=30)
        new_sip = SIP(user_id=user_id, asset_name=asset_name, amount=amount, next_due_date=next_due.strftime('%Y-%m-%d'))
        db.session.add(new_sip)
        
        db.session.commit()
        return jsonify({"success": True, "message": "SIP Activated! First installment processed."}), 201
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/asset_chart', methods=['GET'])
def asset_chart():
    # Generates 30 days of realistic historical chart data for the UI
    chart_data = []
    base = 100.0
    for i in range(30):
        base += random.uniform(-1.5, 2.0)
        chart_data.append(round(base, 2))
    return jsonify({"success": True, "chart": chart_data}), 200

# ==========================================
# 🚀 7. LIVE MARKET DATA (EXACT BULLION MATH)
# ==========================================
@app.route('/api/live_market', methods=['GET'])
def live_market():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        # 1. Scrape Global Futures and Exchange Rate directly from Google Finance
        gold_res = requests.get("https://www.google.com/finance/quote/GCW00:COMEX", headers=headers, timeout=5)
        silver_res = requests.get("https://www.google.com/finance/quote/SIW00:COMEX", headers=headers, timeout=5)
        inr_res = requests.get("https://www.google.com/finance/quote/USD-INR", headers=headers, timeout=5)
        
        gold_soup = BeautifulSoup(gold_res.text, 'html.parser')
        silver_soup = BeautifulSoup(silver_res.text, 'html.parser')
        inr_soup = BeautifulSoup(inr_res.text, 'html.parser')
        
        gold_usd_per_oz = float(gold_soup.find('div', class_='YMlKec fxKbKc').text.replace('$', '').replace(',', ''))
        silver_usd_per_oz = float(silver_soup.find('div', class_='YMlKec fxKbKc').text.replace('$', '').replace(',', ''))
        usd_to_inr = float(inr_soup.find('div', class_='YMlKec fxKbKc').text.replace('₹', '').replace(',', ''))
        
        # 2. Math: 1 Troy Ounce = 31.1035 grams. 
        # Formula: (USD Price / 31.1035) * INR Rate = Exact Physical Price per Gram
        live_gold = round((gold_usd_per_oz / 31.1035) * usd_to_inr, 2)
        live_silver = round((silver_usd_per_oz / 31.1035) * usd_to_inr, 2)
        
        # 3. Generate realistic charts based on the live price
        gold_chart = [round(live_gold * (1 + random.uniform(-0.01, 0.01)), 2) for _ in range(6)] + [live_gold]
        silver_chart = [round(live_silver * (1 + random.uniform(-0.01, 0.01)), 2) for _ in range(6)] + [live_silver]

        return jsonify({
            "success": True,
            "metals": {
                "gold": {"price": f"{live_gold:.2f}", "chart": gold_chart},
                "silver": {"price": f"{live_silver:.2f}", "chart": silver_chart}
            }
        }), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 🚀 8. AI ANALYSIS ENGINE
# ==========================================
@app.route('/api/ai_analysis', methods=['GET'])
@jwt_required()
def ai_analysis():
    try:
        purchases = Purchase.query.filter_by(user_id=get_jwt_identity()).all()
        if not purchases:
            return jsonify({"success": True, "insight": "Your portfolio is empty. Start investing in Nifty 50 or Physical Gold to allow the AI to build your risk profile."}), 200

        total_val = sum((p.buy_price * p.quantity) for p in purchases)
        gold_val = sum((p.buy_price * p.quantity) for p in purchases if '[GOLD]' in p.asset_name or '[SILVER]' in p.asset_name)
        
        gold_pct = (gold_val / total_val) * 100 if total_val > 0 else 0
        
        if gold_pct > 60: 
            insight = f"🚨 AI Alert: Your portfolio is {gold_pct:.1f}% weighted in commodities. Our model suggests pausing Gold SIPs and diversifying into Equities (Stocks/MFs) to capture higher long-term CAGR."
        elif gold_pct < 10: 
            insight = f"🛡️ AI Alert: You lack a strong hedge against inflation. The AI recommends allocating at least 10-15% of your ₹{total_val:,.2f} portfolio to Gold to protect against market crashes."
        else: 
            insight = "✅ AI Alert: Excellent diversification! Your risk-to-equity ratio is mathematically optimized. The AI recommends holding and continuing your current investment strategy."

        return jsonify({"success": True, "insight": insight}), 200
    except Exception as e:
        return jsonify({"success": False, "insight": "AI engine temporarily offline."}), 500

# ==========================================
# 🚀 9. LIVE MARKET NEWS (UNIQUE IMAGES & DAILY UPDATES)
# ==========================================
@app.route('/api/market_news', methods=['GET'])
def market_news():
    try:
        # Google News RSS updates every few minutes!
        url = "https://news.google.com/rss/search?q=indian+stock+market+finance&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        
        news_items = []
        
        # 15 distinct, high-quality finance images
        images = [
            "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1642543492481-44e81e3914a1?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1535320903710-d993d3d77d29?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1579532537598-459ecdaf39cc?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1604594849809-dfedbc827105?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1624996379697-f01d168b1a52?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=400&auto=format&fit=crop",
            "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?q=80&w=400&auto=format&fit=crop"
        ]
        
        # Loop through top 10 articles
        for i, item in enumerate(root.findall('.//item')[:10]):  
            title = item.find('title').text
            link = item.find('link').text
            source_tag = item.find('source')
            source = source_tag.text if source_tag is not None else "Finance News"
            
            # Guarantee a different image for each article systematically
            assigned_image = images[i % len(images)]
            
            news_items.append({
                "title": title,
                "url": link,
                "source": source,
                "image": assigned_image
            })
            
        return jsonify({"success": True, "news": news_items}), 200
        
    except Exception as e:
         return jsonify({"success": False, "news": []}), 500

# ==========================================
# 🚀 10. CRON JOB: BACKGROUND SIP PROCESSOR
# ==========================================
def process_automated_sips():
    """This function runs silently in the background to deduct SIPs"""
    with app.app_context():
        today_str = datetime.now().strftime('%Y-%m-%d')
        # Find all active SIPs due today or earlier
        due_sips = SIP.query.filter(SIP.is_active == True, SIP.next_due_date <= today_str).all()
        
        for sip in due_sips:
            wallet = Wallet.query.filter_by(user_id=sip.user_id).first()
            if wallet and wallet.balance >= sip.amount:
                # Deduct funds and log the purchase
                wallet.balance -= sip.amount
                new_purchase = Purchase(user_id=sip.user_id, asset_name=sip.asset_name, buy_price=sip.amount, quantity=1.0, date=today_str)
                db.session.add(new_purchase)
                
                # Set next deduction date (+30 days)
                next_due = datetime.now() + timedelta(days=30)
                sip.next_due_date = next_due.strftime('%Y-%m-%d')
        
        db.session.commit()

# Start the background cron job to check SIPs every 24 hours
scheduler = BackgroundScheduler()
scheduler.add_job(func=process_automated_sips, trigger="interval", hours=24)
scheduler.start()

# ==========================================
# 🚀 SERVER INITIALIZATION
# ==========================================
# This forces Python to build the tables in your new Render database!
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
