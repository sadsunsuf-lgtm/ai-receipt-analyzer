🧪 Receipt Physician: AI-Powered Financial Health
"You don't have to see the whole staircase, just take the first step."

Receipt Physician is a smart financial analyzer that transforms messy receipt photos into actionable budgeting intelligence using a Hybrid Edge-Cloud Architecture.

🚀 The Innovation
Unlike traditional scanners that send raw (often sensitive) data directly to the cloud, Receipt Physician operates with a privacy-first approach:

Local Edge Processing: Image thresholding, denoising, and OCR are performed locally using OpenCV and Tesseract.

Cloud Intelligence: Only the extracted text is sent to Mistral-7B for high-level financial reasoning and categorization.

🛠️ Tech Stack
Vision: OpenCV (Otsu Thresholding), PIL

OCR: Tesseract OCR

LLM: Mistral-7B-Instruct-v0.2 (via Hugging Face Inference API)

UI: Streamlit & Plotly

📈 Key Features
Resilient Design: Automatic fallback to local mathematical analysis if AI APIs are latent.

Image Optimization: Custom preprocessing pipeline to handle blurry or low-light receipts.

Actionable Insights: Not just data extraction, but personalized "Financial Prescriptions."

Cross-Platform: Smart-path logic allows it to run seamlessly on both Windows (local) and Linux (Streamlit Cloud).
