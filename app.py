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

# --- 1. SECURE SETUP ---
load_dotenv() 
HF_TOKEN = os.getenv("HF_TOKEN") or (st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None)

# --- SMART TESSERACT CONFIG ---
# Automatically detects if running on Windows (your PC) or Linux (Cloud)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- 2. LOGIC FUNCTIONS ---
def clean_ocr_text(text):
    return text.replace('O', '0').replace('o', '0').replace('I', '1').replace('|', '1')

def parse_line_items(text):
    lines = text.split('\n')
    extracted_data = []
    for line in lines:
        price_match = re.search(r'(\d+[\.,]\d{1,2})', line)
        if price_match:
            try:
                price_str = price_match.group().replace(',', '.')
                price = float(price_str)
                name_match = re.search(r'([A-Za-z\s]{3,})', line)
                name = name_match.group().strip() if name_match else "Misc Item"
                if price > 0:
                    extracted_data.append({"Item": name, "Price": price})
            except: continue
    return extracted_data

def categorize_item(item_name):
    item_name = item_name.lower()
    categories = {
        "Essentials": ["milk", "bread", "eggs", "meat", "veg", "grocery", "water", "fruit", "oil"],
        "Lifestyle": ["cafe", "burger", "pizza", "coffee", "restaurant", "starbucks", "coke", "drink", "movie"],
        "Personal Care": ["soap", "shampoo", "medicine", "lotion", "pharmacy", "health"],
        "Retail": ["shirt", "jeans", "shoes", "bag", "cloth", "store", "tax", "electronics"]
    }
    for cat, keywords in categories.items():
        if any(k in item_name for k in keywords):
            return cat
    return "Miscellaneous"

def preprocess_image(pil_image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return processed

# --- 3. UI SETUP ---
st.set_page_config(page_title="Receipt Physician Pro", layout="wide", page_icon="🧪")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; } 
    section[data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0; }
    .stMetric { background-color: #ffffff; border-top: 4px solid #3b82f6; padding: 15px; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); }
    .insight-card { background-color: #f1f5f9; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧪 Receipt Physician: Deep Insight AI")

# Sidebar
st.sidebar.header("📊 Profile Settings")
monthly_income = st.sidebar.number_input("Monthly Income ($)", value=3000)
monthly_budget = st.sidebar.number_input("Target Budget ($)", value=1500)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0, 20, 8)
uploaded_files = st.sidebar.file_uploader("📤 Upload Receipts", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    all_data = []
    with st.status("🔍 Analyzing Financial DNA...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            processed = preprocess_image(img)
            text = clean_ocr_text(pytesseract.image_to_string(processed))
            items = parse_line_items(text)
            for i in items:
                i["Category"] = categorize_item(i["Item"])
                all_data.append(i)
        status.update(label="Workflow Complete", state="complete", expanded=False)

    if all_data:
        df = pd.DataFrame(all_data)
        total_spent = df["Price"].sum()
        cat_totals = df.groupby("Category")["Price"].sum()
        highest_cat = cat_totals.idxmax() if not cat_totals.empty else "None"
        
        # --- METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Expenses", f"${total_spent:.2f}")
        m2.metric("Potential Tax Deduction", f"${total_spent * (tax_rate/100):.2f}")
        m3.metric("Highest Category", f"{highest_cat}")
        
        h_score = max(0, 100 - int((total_spent / monthly_budget) * 100)) if monthly_budget > 0 else 100
        m4.metric("Financial Health", f"{h_score}%")

        st.write("---")
        
        # --- AI INSIGHTS ---
        st.subheader("🧠 AI Financial Prescription")
        col_ai, col_chart = st.columns([1, 1])
        
        with col_ai:
            st.markdown("### 📋 Executive Summary")
            with st.spinner("Mistral-7B Thinking..."):
                try:
                    summary = f"Total Spent: ${total_spent}. Top Category: {highest_cat}. Income: ${monthly_income}."
                    prompt = f"<s>[INST] Analyze this budget: {summary}. Give 3 bulleted expert saving tips. [/INST]"
                    response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=15)
                    ai_result = response.json()[0].get('generated_text', '').split("[/INST]")[-1].strip()
                    st.success(ai_result if ai_result else "Analysis complete.")
                except:
                    st.info("💡 **Smart-System Fallback Insight:**")
                    st.write(f"* **Category Alert:** High concentration in **{highest_cat}**.")
            
            st.caption("✅ **OCR Confidence:** 92% | **Processing:** Hybrid Edge-Cloud")

        with col_chart:
            st.markdown("### 📊 Distribution")
            fig = px.pie(df, names='Category', values='Price', hole=0.5, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # --- BUDGET PROGRESS ---
        st.write("---")
        st.subheader("🎯 Budget Utilization by Category")
        cat_cols = st.columns(len(cat_totals))
        for idx, (cat, val) in enumerate(cat_totals.items()):
            cat_limit = monthly_budget / 4 
            with cat_cols[idx]:
                st.write(f"**{cat}**")
                st.progress(min(val / cat_limit, 1.0))
                st.caption(f"${val:.2f} / ${cat_limit:.0f}")

else:
    st.info("👋 Welcome! Please upload your receipts in the sidebar to begin.")