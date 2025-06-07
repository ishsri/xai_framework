# xai_methods/smoothgradpp.py

from captum.attr import NoiseTunnel, IntegratedGradients

def apply_smoothgradpp(model, input_tensor, target_class):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr = nt.attribute(input_tensor,
                        nt_type='smoothgrad_sq',
                        n_samples=20,
                        target=target_class)
    return attr.squeeze().detach().numpy()
