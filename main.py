import os
import streamlit as st
from groq import Groq
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize services
@st.cache_resource
def init_services():
    return {
        'llm': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': YOLO('yolov8n.pt')  # Vehicle detection model
    }

services = init_services()

# Vehicle class mapping (YOLOv8 default)
VEHICLE_CLASSES = {
    2: 'car', 5: 'bus', 7: 'truck', 
    3: 'motorcycle', 6: 'train'
}

def analyze_traffic(image):
    """Process image using pure PIL"""
    # Run detection directly on PIL Image
    results = services['model'](image)
    
    # Count vehicles and generate annotation
    counts = {v:0 for v in VEHICLE_CLASSES.values()}
    annotated_img = image.copy()
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        if class_id in VEHICLE_CLASSES:
            counts[VEHICLE_CLASSES[class_id]] += 1
    
    return counts, results[0].plot()  # plot() returns PIL Image

def generate_report(counts):
    """Generate traffic analysis summary"""
    total_vehicles = sum(counts.values())
    prompt = f"""
    Analyze this traffic scene:
    - Vehicles detected: {total_vehicles}
    - Breakdown: {', '.join(f'{k}:{v}' for k,v in counts.items() if v > 0)}
    
    Provide a concise report including:
    1. Traffic density assessment
    2. Dominant vehicle types
    3. Potential bottlenecks
    """
    
    response = services['llm'].chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a traffic analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("🚦 Traffic Flow Analyzer")
uploaded_file = st.file_uploader(
    "Upload traffic image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    with st.spinner("Analyzing traffic..."):
        counts, annotated_img = analyze_traffic(image)
        report = generate_report(counts)
        
        with col2:
            st.image(annotated_img, caption="Analysis Results", use_container_width=True)
        
        st.subheader("Traffic Report")
        st.write(report)
