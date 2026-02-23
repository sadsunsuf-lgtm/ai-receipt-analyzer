import os
import re
import cv2
import numpy as np
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# --- 1. SETUP & SECURE KEYS ---
load_dotenv() 
# Ensure your .env file has HF_TOKEN=your_token_here
HF_TOKEN = os.getenv("HF_TOKEN") or (st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Using Mistral-7B - High reliability model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- 2. ENHANCED CATEGORIZATION ---
def categorize_item(item_name):
    name = item_name.lower().strip()
    
    categories = {
        "Sugar & Sweets": ["sugar", "honey", "syrup", "candy", "jam", "jelly", "sweet", "dessert", "mithai"],
        "Dairy & Eggs": ["milk", "cheese", "yogurt", "butter", "egg", "curd", "paneer", "dairy"],
        "Bakery & Grains": ["bread", "flour", "rice", "wheat", "pasta", "cake", "bisc", "toast", "cereal", "ata"],
        "Meat & Seafood": ["chicken", "beef", "fish", "meat", "mutton", "shrimp", "steak", "nugget"],
        "Fruits & Vegetables": ["apple", "banana", "potato", "onion", "tomato", "veg", "fruit", "garlic", "salad"],
        "Snacks & Drinks": ["chips", "snack", "choc", "coke", "pepsi", "juice", "water", "coffee", "tea", "ice"],
        "Household & Care": ["soap", "shamp", "deter", "clean", "tiss", "brush", "wash", "health", "surf", "lux"],
        "Lifestyle & Tax": ["tax", "fee", "service", "rs", "vat", "tui", "bag", "total", "amount", "payable", "sub", "ttal"]
    }
    
    for cat, keywords in categories.items():
        if any(k in name for k in keywords):
            return cat
    return "Miscellaneous"

def parse_line_items(text):
    lines = text.split('\n')
    extracted_data = []
    for line in lines:
        # Regex to find price - improved to catch decimals correctly
        price_match = re.search(r'(\d+[\.,]\d{2}|\d+\.\d{1,2}| \d{2,5}$)', line)
        if price_match:
            try:
                # ONLY convert 'o' to '0' inside the price area for OCR correction
                raw_price = price_match.group().strip().replace('o', '0').replace('O', '0').replace(',', '.')
                price = float(raw_price)
                
                # Item Name stays original (keeping 'o' intact)
                item_name = line.replace(price_match.group(), "").strip()
                item_name = re.sub(r'[^a-zA-Z\s]', '', item_name).strip()
                
                if len(item_name) >= 2 and price > 0.1:
                    category = categorize_item(item_name)
                    # Filter out subtotals from being treated as items
                    if any(x in item_name.lower() for x in ["total", "payable", "amount", "sub", "ttal"]):
                        category = "Lifestyle & Tax"

                    extracted_data.append({"Item Name": item_name, "Price": price, "Category": category})
            except: continue
    return extracted_data

def preprocess_image(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return processed

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="Receipt Physician Pro", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; } 
    .stMetric { background-color: #f8fafc; border-top: 4px solid #3b82f6; padding: 15px; border-radius: 12px; }
    .ai-box { background-color: #f0f7ff; border: 1px solid #3b82f6; padding: 25px; border-radius: 15px; color: #1e3a8a; line-height: 1.6; }
    .report-header { color: #1e3a8a; border-bottom: 2px solid #3b82f6; padding-bottom: 5px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Receipt Physician: Elite Financial Dashboard")

# Sidebar
st.sidebar.header("📊 Financial Profile")
income = st.sidebar.number_input("Monthly Income ($)", value=3000.0)
budget = st.sidebar.number_input("Target Budget ($)", value=1500.0)
uploaded_files = st.sidebar.file_uploader("📤 Upload Receipts", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    all_items = []
    with st.status("🔍 Analyzing Financial DNA...", expanded=False):
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            processed = preprocess_image(img)
            raw_text = pytesseract.image_to_string(processed)
            all_items.extend(parse_line_items(raw_text))

    if all_items:
        df = pd.DataFrame(all_items)
        total_spent = df["Price"].sum()
        cat_totals = df.groupby("Category")["Price"].sum().sort_values(ascending=False)
        top_cat = cat_totals.index[0]
        usage_pct = (total_spent / budget) * 100
        
        # --- METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Spend", f"${total_spent:,.2f}")
        m2.metric("Remaining", f"${budget - total_spent:,.2f}")
        m3.metric("Top Category", top_cat)
        m4.metric("Budget Usage", f"{int(usage_pct)}%")

        # --- DETAILED AI REPORT ---
        st.subheader("👨‍💼 Senior Financial Consultant: Strategic Audit")
        
        report_container = st.empty()
        
        with st.spinner("Generating Professional Report..."):
            ai_success = False
            if HF_TOKEN:
                try:
                    prompt = f"""<s>[INST] You are a Senior Financial Consultant. 
                    Act strictly as a professional auditor. Analyze this data:
                    Spend: ${total_spent} | Income: ${income} | Budget: ${budget}
                    Top Category: {top_cat} (${cat_totals.iloc[0]})
                    Breakdown: {cat_totals.to_dict()}
                    
                    Structure your response exactly like this:
                    ### 📊 EXECUTIVE HEALTH CHECK
                    (Analyze the burn rate and income-to-spend ratio)
                    ### 🔎 EXPENDITURE ANOMALIES
                    (Identify risks in specific categories)
                    ### 📉 OPTIMIZATION STRATEGY
                    (3 high-impact steps to save money)
                    ### 💡 VERDICT
                    (Final professional recommendation)
                    [/INST]"""
                    
                    response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 1000, "temperature": 0.7}}, timeout=25)
                    result = response.json()
                    
                    if isinstance(result, list) and 'generated_text' in result[0]:
                        ai_text = result[0]['generated_text'].split("[/INST]")[-1].strip()
                        report_container.markdown(f"<div class='ai-box'>{ai_text}</div>", unsafe_allow_html=True)
                        ai_success = True
                except Exception as e:
                    ai_success = False

            # --- SMART FALLBACK (Runs if AI Fails or No Token) ---
            if not ai_success:
                status_color = "🔴 CRITICAL" if usage_pct > 90 else "🟡 CAUTION" if usage_pct > 70 else "🟢 HEALTHY"
                fallback_report = f"""
                <div class='ai-box'>
                <h3 class='report-header'>📊 EXECUTIVE HEALTH CHECK</h3>
                Current Status: <b>{status_color}</b><br>
                You have consumed <b>{usage_pct:.1f}%</b> of your monthly budget. 
                Based on your income of ${income:,.2f}, your current savings rate is <b>{((income-total_spent)/income)*100:.1f}%</b>.
                
                <h3 class='report-header'>🔎 EXPENDITURE ANOMALIES</h3>
                Your spending is heavily concentrated in <b>{top_cat}</b>, accounting for <b>{(cat_totals.iloc[0]/total_spent)*100:.1f}%</b> of total outlay. 
                Any single category (excluding rent) exceeding 30% of total spend is flagged as a "Budget Leak."
                
                <h3 class='report-header'>📉 OPTIMIZATION STRATEGY</h3>
                1. <b>The 10% Pivot:</b> Reducing <b>{top_cat}</b> by just 10% would save you <b>${cat_totals.iloc[0]*0.1:,.2f}</b> immediately.<br>
                2. <b>Tax Shield:</b> Ensure all items in 'Lifestyle & Tax' are documented for potential end-of-year deductions.<br>
                3. <b>Surplus Allocation:</b> Redirect your current ${income - total_spent:,.2f} surplus into a high-yield vehicle.
                
                <h3 class='report-header'>💡 VERDICT</h3>
                The portfolio is currently <b>{'Stable' if total_spent < budget else 'Volatile'}</b>. 
                Immediate attention to <b>{top_cat}</b> is required to maintain the year-end savings target.
                </div>
                """
                report_container.markdown(fallback_report, unsafe_allow_html=True)

        # --- CHARTS ---
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df, names='Category', values='Price', hole=0.5, 
                                 title="Spending Weight by Department", 
                                 color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(cat_totals, title="Absolute Department Spend", 
                                  color=cat_totals.index,
                                  labels={'value':'Total Price ($)', 'Category':'Department'}), use_container_width=True)

        # --- DATA TABLE ---
        st.write("---")
        st.subheader("📋 Verified Itemization Trail")
        st.dataframe(df, use_container_width=True)

else:
    st.info("👋 Welcome! Please upload your receipts in the sidebar to begin your end-to-end financial analysis.")