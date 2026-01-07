import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

@st.cache_resource
def load_model():
    model = torch.jit.load("brain_tumor_model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class_names = ["No Tumor", "Tumor"]

st.title("Brain Tumor Detection")

file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    st.write(f"Prediction: **{class_names[pred.item()]}**")
    st.write(f"Confidence: **{conf.item() * 100:.2f}%**")
