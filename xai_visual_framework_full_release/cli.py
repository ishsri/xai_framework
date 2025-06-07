# cli.py

import typer
from main import run_xai
from visualize import plot_composite
from evaluator import deletion_score, plot_deletion_curves
from xai_methods import utils

app = typer.Typer()

@app.command()
def run(image: str,
        model_path: str = None,
        arch: str = "resnet50",
        layer: str = "layer4",
        target_class: int = 243,
        methods: list[str] = ["grad_cam", "occlusion"]):
    
    results = run_xai(image, model_path, arch, layer, target_class, methods)
    plot_composite(results, image)

    # Evaluation
    model = utils.load_model(arch, model_path)
    input_tensor = utils.preprocess_image(image)
    eval_scores = {}
    for method, attr in results.items():
        x, y, score = deletion_score(input_tensor, attr, model, target_class)
        eval_scores[method] = (x, y, score)
    
    plot_deletion_curves(eval_scores)

if __name__ == "__main__":
    app()
