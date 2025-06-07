# streamlit_app.py

import streamlit as st
from main import run_xai
from visualize import plot_composite
from evaluator import deletion_score, plot_deletion_curves
from xai_methods import utils
import os
import tempfile

st.set_page_config(page_title="XAI Medical Visualizer", layout="wide")
st.title("🩺 Explainable AI Visual Dashboard for Medical Imaging")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])
model_path = st.text_input("Path to pretrained model (leave blank to use torchvision default)", "")
arch = st.selectbox("Model architecture", ["resnet50", "densenet121"])
target_class = st.number_input("Target class index", value=243, min_value=0, step=1)
layer = st.text_input("Layer name for Grad-CAM (e.g., layer4)", "layer4")

methods = st.multiselect("XAI methods to run", [
    "grad_cam", "occlusion", "integrated_gradients", "lrp", "smoothgradpp", "guided_backprop"
], default=["grad_cam"])

if st.button("🔍 Run XAI Analysis"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    with st.spinner("Running..."):
        results = run_xai(image_path, model_path if model_path else None, arch, layer, target_class, methods)
        plot_composite(results, image_path)
        st.image("composite_output.png", caption="Composite XAI Map", use_column_width=True)

        model = utils.load_model(arch, model_path if model_path else None)
        input_tensor = utils.preprocess_image(image_path)
        eval_scores = {}
        for method, attr in results.items():
            x, y, score = deletion_score(input_tensor, attr, model, target_class)
            eval_scores[method] = (x, y, score)

        plot_deletion_curves(eval_scores)
        st.image("deletion_curves.png", caption="Deletion Score Curves", use_column_width=True)

    os.remove(image_path)
