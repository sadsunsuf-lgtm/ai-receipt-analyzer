import streamlit as st
import pandas as pd
import plotly.express as px
from app import preprocess_receipt, get_hf_insight # Import from your existing logic
import pytesseract
import re
from PIL import Image

st.set_page_config(page_title="Receipt Physician", page_icon="🧪")

st.title("🧪 Receipt Physician")
st.markdown("*Answering the 'Where did my money go?' question with Hybrid AI.*")

uploaded_file = st.file_uploader("Upload a receipt (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Receipt", use_container_width=True)
    
    with st.spinner("Processing with Hybrid OCR Engine..."):
        # Save temp file
        with open("temp_receipt.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 1. Image Processing
        processed_img = preprocess_receipt("temp_receipt.jpg")
        raw_text = pytesseract.image_to_string(processed_img)
        
        # 2. Data Parsing
        prices = [float(p) for p in re.findall(r'\d+\.\d{2}', raw_text)]
        
        # 3. Visuals
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detected", f"${sum(prices):.2f}")
        with col2:
            st.metric("Items Found", len(prices))

        # 4. AI Insight
        st.subheader("🧠 AI Financial Advice")
        advice = get_hf_insight(raw_text, prices)
        st.info(advice)

        # 5. Charts
        if prices:
            df = pd.DataFrame({"Item": [f"Item {i+1}" for i in range(len(prices))], "Price": prices})
            fig = px.pie(df, values='Price', names='Item', title='Spending Breakdown')
            st.plotly_chart(fig)