# xai_methods/guided_backprop.py

from captum.attr import GuidedBackprop

def apply_guided_backprop(model, input_tensor, target_class):
    gbp = GuidedBackprop(model)
    attr = gbp.attribute(input_tensor, target=target_class)
    return attr.squeeze().detach().numpy()
