import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return "Image not found"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Simple threshold for a quick look
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return pytesseract.image_to_string(thresh)