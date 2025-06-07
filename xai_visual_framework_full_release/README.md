# 🧠 XAI Visual Framework for Medical Imaging

This project generates side-by-side explainability maps (Grad-CAM, LRP, IG, etc.) and evaluates them using deletion scores. Supports both CLI and Streamlit dashboard.

## Features
- ✅ Grad-CAM, Occlusion, IG, LRP, SmoothGrad++, Guided Backprop
- ✅ Model-agnostic PyTorch support
- ✅ CLI interface with Typer
- ✅ Streamlit web dashboard
- ✅ Evaluation curves (AUC + deletion)

## Quickstart (CLI)
```bash
python cli.py run --image sample.jpg --arch resnet50 --target-class 243 --methods grad_cam lrp
