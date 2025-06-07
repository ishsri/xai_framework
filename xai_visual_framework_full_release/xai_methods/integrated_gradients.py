# xai_methods/integrated_gradients.py

from captum.attr import IntegratedGradients

def apply_ig(model, input_tensor, target_class):
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_tensor, target=target_class, n_steps=50)
    return attr.squeeze().detach().numpy()
