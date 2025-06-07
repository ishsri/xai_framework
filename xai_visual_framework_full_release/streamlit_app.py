# streamlit_app.py

import streamlit as st
from main import run_xai
from visualize import plot_composite
from evaluator import deletion_score, plot_deletion_curves
from xai_methods import utils
import os
import tempfile
from PIL import Image

st.set_page_config(page_title="XAI for Haematology", layout="wide")
st.title("🔬 Explainable AI for Bone Marrow / Haematology Imaging")

# --- Workflow selection ---
mode = st.radio("Choose Mode", ["Pretrained Workflow", "Custom Upload Workflow"])

# --- Pretrained Workflow ---
if mode == "Pretrained Workflow":
    st.subheader("Select sample image(s) for XAI visualization")
    
    SAMPLE_FOLDER = "pretrained_samples"
    sample_images = sorted([f for f in os.listdir(SAMPLE_FOLDER) if f.endswith((".jpg", ".png"))])
    
    selected_images = st.multiselect("Choose image(s)", sample_images, default=sample_images[:1])
    
    pretrained_models = {
        "ResNet50 (bone marrow pretrained)": "models/resnet50_bm.pt",
        "DenseNet121 (RBC fine-tuned)": "models/densenet121_rbc.pt"
    }
    model_choice = st.selectbox("Choose Pretrained Model", list(pretrained_models.keys()))
    target_class = st.number_input("Target class index", value=1, min_value=0)
    layer = st.text_input("Layer (for Grad-CAM)", "layer4")
    
    methods = st.multiselect("XAI methods", [
        "grad_cam", "occlusion", "integrated_gradients", "lrp", "smoothgradpp", "guided_backprop"
    ], default=["grad_cam"])

    if st.button("🧪 Run Pretrained XAI"):
        with st.spinner("Processing..."):
            for img_name in selected_images:
                img_path = os.path.join(SAMPLE_FOLDER, img_name)
                results = run_xai(img_path, pretrained_models[model_choice], "resnet50", layer, target_class, methods)
                plot_composite(results, img_path, save_path=f"composite_{img_name}")
                st.image(f"composite_{img_name}", caption=f"{img_name} - Composite Heatmap", use_column_width=True)
        
# --- Custom Upload Workflow ---
elif mode == "Custom Upload Workflow":
    st.subheader("Upload your own image and model")
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "png"])
    uploaded_model = st.file_uploader("Upload model (.pt)", type=["pt", "pth"])
    
    arch = st.selectbox("Model architecture", ["resnet50", "densenet121"])
    target_class = st.number_input("Target class index", value=243, min_value=0)
    layer = st.text_input("Layer name for Grad-CAM", "layer4")
    
    methods = st.multiselect("XAI methods", [
        "grad_cam", "occlusion", "integrated_gradients", "lrp", "smoothgradpp", "guided_backprop"
    ], default=["grad_cam"])

    if st.button("🔍 Run XAI on Custom Image"):
        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(uploaded_image.read())
                image_path = tmp_img.name

            model_path = None
            if uploaded_model:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
                    tmp_model.write(uploaded_model.read())
                    model_path = tmp_model.name

            with st.spinner("Running..."):
                results = run_xai(image_path, model_path, arch, layer, target_class, methods)
                plot_composite(results, image_path)
                st.image("composite_output.png", caption="Composite XAI", use_column_width=True)

                model = utils.load_model(arch, model_path)
                input_tensor = utils.preprocess_image(image_path)
                eval_scores = {}
                for method, attr in results.items():
                    x, y, score = deletion_score(input_tensor, attr, model, target_class)
                    eval_scores[method] = (x, y, score)

                plot_deletion_curves(eval_scores)
                st.image("deletion_curves.png", caption="Deletion Score Curves", use_column_width=True)

                os.remove(image_path)
                if model_path:
                    os.remove(model_path)
        else:
            st.warning("Please upload an image.")

