import streamlit as st
import torch
import os
import numpy as np
from PIL import Image
from xai_methods.utils import load_model, preprocess_image
from xai_methods.grad_cam import grad_cam
from xai_methods.occlusion import occlusion_map
from xai_methods.integrated_gradients import integrated_gradients_map
from xai_methods.lrp import lrp_map
from xai_methods.score_cam import score_cam
from xai_methods.smoothgradpp import smoothgradpp_map
from xai_methods.guided_backprop import guided_backprop_map
from evaluator import deletion_score, insertion_score, compute_auc
from visualize import plot_composite
from exporter import export_html
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile

METHOD_MAP = {
    "Grad-CAM": grad_cam,
    "Occlusion": occlusion_map,
    "Integrated Gradients": integrated_gradients_map,
    "LRP": lrp_map,
    "Score-CAM": score_cam,
    "SmoothGrad++": smoothgradpp_map,
    "Guided Backprop": guided_backprop_map,
}

st.set_page_config(page_title="XAI Medical Dashboard", layout="wide")

st.sidebar.title("XAI Settings")
uploaded_image = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
target_class = st.sidebar.number_input("Target Class Index", min_value=0, value=243)
arch = st.sidebar.selectbox("Model Architecture", ["resnet18", "resnet50"])
layer = st.sidebar.text_input("Target Layer", value="layer4")
methods_selected = st.sidebar.multiselect("Select XAI Methods", list(METHOD_MAP.keys()), default=list(METHOD_MAP.keys()))
run_button = st.sidebar.button("Run XAI Analysis")

if run_button and uploaded_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_image.read())
        tmp_path = tmp.name

    st.success("Running selected XAI methods...")
    model = load_model(arch_name=arch)
    input_tensor, raw_image = preprocess_image(tmp_path)

    maps, names, del_scores, ins_scores, auc_results = [], [], [], [], []
    for name in methods_selected:
        method_fn = METHOD_MAP[name]
        if "cam" in name.lower():
            result = method_fn(model, input_tensor, target_class, layer_name=layer)
        else:
            result = method_fn(model, input_tensor, target_class)
        maps.append(result)
        names.append(name)

        # Evaluation metrics
        t_attrib = torch.tensor(result).unsqueeze(0).unsqueeze(0)
        del_curve = deletion_score(model, input_tensor, t_attrib, target_class)
        ins_curve = insertion_score(model, input_tensor, t_attrib, target_class)
        del_scores.append(del_curve)
        ins_scores.append(ins_curve)
        auc_results.append({
            "method": name,
            "deletion_auc": compute_auc(del_curve),
            "insertion_auc": compute_auc(ins_curve)
        })

    # Composite plot
    composite_path = "composite_output.png"
    plot_composite(raw_image, maps, names, composite_path)
    st.image(composite_path, caption="Composite XAI Heatmaps", use_column_width=True)

    # Plotly curves
    st.subheader("Deletion & Insertion Curves")
    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(y=del_scores[i], mode='lines', name=f"{name} - Deletion"))
        fig.add_trace(go.Scatter(y=ins_scores[i], mode='lines', name=f"{name} - Insertion"))
    st.plotly_chart(fig, use_container_width=True)

    # AUC table
    st.subheader("AUC Scores")
    st.table({x['method']: {"Deletion AUC": round(x['deletion_auc'], 3), "Insertion AUC": round(x['insertion_auc'], 3)} for x in auc_results})

    # Export section
    if st.button("Export HTML Report"):
        export_html(names, del_scores, filename="report.html")
        st.success("HTML report saved as report.html")

    os.remove(tmp_path)