import os
import streamlit as st
from groq import Groq
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def init_services():
    return {
        'llm': Groq(api_key=os.getenv("GROQ_API_KEY")),
        'model': YOLO('yolov8n.pt')  # General purpose model
    }

services = init_services()

# Vehicle classes + common non-traffic objects
RELEVANT_CLASSES = {
    2: 'car', 5: 'bus', 7: 'truck', 3: 'motorcycle',
    0: 'person', 16: 'dog', 56: 'chair', 64: 'potted plant'
}

def analyze_image(image):
    """Enhanced detection with context awareness"""
    results = services['model'](image)
    counts = {v:0 for v in RELEVANT_CLASSES.values()}
    detected_objects = []
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        if class_id in RELEVANT_CLASSES:
            obj_name = RELEVANT_CLASSES[class_id]
            counts[obj_name] += 1
            detected_objects.append(obj_name)
    
    return counts, results[0].plot(), detected_objects

def generate_report(counts, detected_objects):
    """Context-aware report generation"""
    total_vehicles = sum(counts[k] for k in ['car', 'bus', 'truck', 'motorcycle'])
    total_objects = len(detected_objects)
    
    # Situation analysis
    if total_vehicles == 0:
        if total_objects == 0:
            situation = "EMPTY_IMAGE"
        elif any(obj in detected_objects for obj in ['person', 'dog', 'chair']):
            situation = "NON_TRAFFIC"
        else:
            situation = "EMPTY_ROAD"
    else:
        situation = "TRAFFIC"
    
    # Generate appropriate prompt
    if situation == "EMPTY_IMAGE":
        return "No objects detected in the uploaded image. Please upload a clearer image."
    
    elif situation == "EMPTY_ROAD":
        prompt = f"""
        This image shows a road with no traffic. 
        Detected objects: {', '.join(set(detected_objects)) or 'none'}
        Provide 2-3 brief observations about the road condition.
        """
    
    elif situation == "NON_TRAFFIC":
        prompt = f"""
        This image doesn't contain traffic but shows: 
        {', '.join(set(detected_objects))}
        Generate a friendly note explaining this isn't a traffic image,
        along with 1 interesting observation about the content.
        """
    
    else:  # TRAFFIC
        prompt = f"""
        Analyze this traffic scene:
        - Vehicles: {total_vehicles}
        - Breakdown: {', '.join(f'{k}:{v}' for k,v in counts.items() if v > 0 and k in ['car', 'bus', 'truck', 'motorcycle'])}
        Provide:
        1. Density assessment
        2. Dominant vehicle types
        3. One safety observation
        """
    
    response = services['llm'].chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a versatile image analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Traffic Scene Analyzer")
uploaded_file = st.file_uploader("Upload any image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    counts, result_img, detected_objects = analyze_image(image)
    
    col1, col2 = st.columns(2)
    col1.image(image, use_container_width=True, caption="Original Image")
    col2.image(result_img, use_container_width=True, caption="Analysis Results")
    
    report = generate_report(counts, detected_objects)
    st.subheader("Analysis Report")
    st.write(report)
    
    if sum(counts.values()) > 0:
        st.subheader("Detected Objects")
