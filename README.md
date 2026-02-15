🧪 Receipt Physician: AI-Powered Financial Health
"You don't have to see the whole staircase, just take the first step."

🚀 The Innovation
Most receipt scanners rely 100% on the cloud. Receipt Physician uses a Hybrid Edge-Cloud Architecture. We perform image thresholding and OCR locally to ensure data integrity, then use Mistral-7B for high-level financial reasoning.

🛠️ Tech Stack
Vision: OpenCV (Otsu Thresholding), PIL

OCR: Tesseract OCR v5

LLM: Mistral-7B-Instruct (via Hugging Face)

UI: Streamlit & Plotly

📈 Key Features
Resilient Design: Automatic fallback to local mathematical analysis if AI APIs are latent.

Image Optimization: Custom preprocessing pipeline to handle blurry or low-light receipts.

Actionable Insights: Not just data extraction, but personalized budgeting tips.