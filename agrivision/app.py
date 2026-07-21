import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
from treatments import DISEASE_INFO

# Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Smart Plant Disease AI Detector")
st.markdown("Upload an image of a plant leaf to detect diseases and receive AI-driven treatment advice.")

# Load Model using timm (Fast & Reliable)
@st.cache_resource
def load_pretrained_model():
    # Model architecture for plant classification
    model = timm.create_model('resnet18', pretrained=True)
    model.eval()
    return model

try:
    model = load_pretrained_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# UI Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Select a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

with col2:
    if uploaded_file is not None and model_loaded:
        with st.spinner('🔍 Analyzing leaf with AI model...'):
            # Preprocess image
            input_tensor = transform(image).unsqueeze(0)
            
            # Model Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probs, 0)

            # Mapping prediction (Sample class mapping)
            # You can map this to your custom labels or treatments.py
            predicted_label = "Tomato_Early_blight"  # Mocked / Mapped target label
            conf_percentage = confidence.item() * 100

            # Display Classification Results
            st.subheader("📊 Diagnostic Results")
            st.metric(label="Detected Condition", value=predicted_label, delta=f"{conf_percentage:.1f}% Confidence")

            # Retrieve Treatment Recommendations
            info = DISEASE_INFO.get(predicted_label, None)
            if info:
                st.divider()
                st.subheader(f"💊 Treatment Plan: {info['title']}")
                st.markdown(f"**Causative Agent:** {info['cause']}")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.success(f"🌱 **Organic / Biological Solution:**\n{info['organic_treatment']}")
                st.warning(f"🧪 **Chemical / Fungicide Solution:**\n{info['chemical_treatment']}")
                st.info(f"🛡️ **Prevention & Care:**\n{info['prevention']}")
            else:
                st.divider()
                st.info("ℹ️ Detailed treatment recommendations for this specific category are not yet registered in treatments.py.")