# xai_methods/occlusion.py

from captum.attr import Occlusion

def apply_occlusion(model, input_tensor, target_class):
    occlusion = Occlusion(model)
    attr = occlusion.attribute(input_tensor,
                               strides=(1, 8, 8),
                               sliding_window_shapes=(3, 15, 15),
                               target=target_class)
    return attr.squeeze().detach().numpy()
