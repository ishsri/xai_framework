# xai_methods/lrp.py

from captum.attr import LRP

def apply_lrp(model, input_tensor, target_class):
    lrp = LRP(model)
    attr = lrp.attribute(input_tensor, target=target_class)
    return attr.squeeze().detach().numpy()
