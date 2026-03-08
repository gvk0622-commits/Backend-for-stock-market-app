from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
import os
import random
import yfinance as yf
from mftool import Mftool
import requests
from bs4 import BeautifulSoup
from datetime import timedelta

app = Flask(__name__)
CORS(app) # Allows your Flutter app to talk to Python securely
@app.route('/')
def home():
    return "🚀 Kasu Backend API is LIVE and running smoothly on Docker!"
# ==========================================
# 🚀 1. SECURE DATABASE CONFIGURATION
# ==========================================
# Render provides the DATABASE_URL environment variable automatically
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
# 🚀 6. AI ANALYSIS ENGINE
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
# 🚀 7. LIVE MARKET DATA SCRAPING (GOLD & SILVER)
# ==========================================
@app.route('/api/live_market', methods=['GET'])
def live_market():
    # 1. Scrape Live Gold/Silver Prices from GoodReturns (or use yfinance fallbacks)
    gold_price = "7450.00"
    silver_price = "85.40"
    gold_chart = []
    silver_chart = []

    try:
        # Fetching Gold ETF and Silver ETF from Yahoo Finance as a reliable proxy for physical charts
        gold_ticker = yf.Ticker("GOLDBEES.NS")
        silver_ticker = yf.Ticker("SILVERBEES.NS")
        
        g_hist = gold_ticker.history(period="7d")
        s_hist = silver_ticker.history(period="7d")
        
        # We use the ETF percentage changes to generate realistic charts
        if not g_hist.empty:
            gold_chart = g_hist['Close'].tolist()
            # Try to scrape physical gold price, fallback to ETF * multiplier
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                res = requests.get('https://www.goodreturns.in/gold-rates/india.html', headers=headers, timeout=3)
                soup = BeautifulSoup(res.text, 'html.parser')
                # Find the 22K 1g price block
                price_block = soup.find('div', class_='gold_silver_table').find('strong', id='el')
                if price_block:
                     gold_price = price_block.text.replace('₹', '').replace(',', '').strip()
            except:
                gold_price = str(round(gold_chart[-1] * 120, 2)) # Fallback multiplier
                
        if not s_hist.empty:
             silver_chart = s_hist['Close'].tolist()
             # Scrape silver
             try:
                 res = requests.get('https://www.goodreturns.in/silver-rates/india.html', headers=headers, timeout=3)
                 soup = BeautifulSoup(res.text, 'html.parser')
                 price_block = soup.find('div', class_='gold_silver_table').find('strong', id='el')
                 if price_block:
                     # Usually per KG, convert to per Gram
                     kg_price = float(price_block.text.replace('₹', '').replace(',', '').strip())
                     silver_price = str(round(kg_price / 1000, 2))
             except:
                 silver_price = str(round(silver_chart[-1], 2))

    except Exception as e:
        print(f"Scraping error: {e}")
        # Fallback data is automatically used by Flutter if empty arrays are sent

    return jsonify({
        "success": True,
        "metals": {
            "gold": {
                "price": gold_price,
                "chart": gold_chart
            },
            "silver": {
                "price": silver_price,
                "chart": silver_chart
            }
        },
        "etfs": {
            "NIFTYBEES": "265.40", # You can replace with live yf data later
            "GOLDBEES": "64.20"
        }
    }), 200

# ==========================================
# 🚀 8. LIVE EQUITY & MF DATA (NEW)
# ==========================================
@app.route('/api/live_equities', methods=['GET'])
def live_equities():
    try:
        # 1. Fetch live stock data using yfinance
        stock_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
        live_stocks = []
        
        for t in stock_tickers:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="2d") # Get yesterday and today
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                live_stocks.append({
                    "symbol": t.replace('.NS', ''),
                    "price": f"₹{current_price:,.2f}",
                    "change": f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%",
                    "isUp": str(change_pct >= 0).lower()
                })

        # 2. Fetch live MF data using mftool
        # Note: AMFI codes are needed here. Example: 120503 is SBI Small Cap
        mf_codes = ["120503", "120504", "118272"] 
        live_mfs = []
        
        for code in mf_codes:
            nav_data = mf.get_scheme_quote(code)
            if nav_data:
                live_mfs.append({
                    "name": nav_data['scheme_name'],
                    "price": f"₹{nav_data['nav']}",
                    "date": nav_data['date']
                })

        return jsonify({
            "success": True,
            "stocks": live_stocks,
            "mutual_funds": live_mfs
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==========================================
# 🚀 9. LIVE MARKET NEWS (WITH IMAGES)
# ==========================================
@app.route('/api/market_news', methods=['GET'])
def market_news():
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        # Find article blocks
        articles = soup.find_all('li', class_='clearfix', limit=5) 
        
        for article in articles:
            a_tag = article.find('a')
            img_tag = article.find('img') # 🚀 Extract the image tag
            
            if a_tag and a_tag.get('title'):
                # Extract the image URL, provide a fallback if missing
                image_url = ""
                if img_tag and img_tag.get('data-src'):
                    image_url = img_tag.get('data-src')
                elif img_tag and img_tag.get('src'):
                    image_url = img_tag.get('src')
                else:
                    # Provide a generic stock market image fallback
                    image_url = "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=200&auto=format&fit=crop"

                news_items.append({
                    "title": a_tag.get('title'),
                    "url": a_tag.get('href'),
                    "source": "Moneycontrol",
                    "image": image_url # 🚀 Add the image to the response
                })
                
        if not news_items:
             news_items = [
                {"title": "Markets hit new highs amid strong Q3 earnings", "url": "#", "source": "Finance Weekly", "image": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=200&auto=format&fit=crop"},
             ]

        return jsonify({"success": True, "news": news_items}), 200
        
    except Exception as e:
         return jsonify({
             "success": False, 
             "news": [
                {"title": "Markets hit new highs amid strong Q3 earnings", "url": "#", "source": "Finance Weekly", "image": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=200&auto=format&fit=crop"}
             ]
         }), 200


# ==========================================
# 🚀 SERVER INITIALIZATION
# ==========================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


