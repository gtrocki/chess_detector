import torch


########################--SCALING--############################
def scale_to_0_1(tensor):
    """
    Scale the values of a tensor to the range [0, 1].

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Scaled tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()

    scaled_tensor = (tensor - min_val) / (max_val - min_val)

    return scaled_tensor


class ScalingTransform:
    def __call__(self, tensor):
        # Apply your custom transformation here
        # For example, resize the image
        scaled_tensor = scale_to_0_1(tensor)
        return scaled_tensor
###############################################################
########################--PERMUTE--############################
class PermuteChannelIndex:
     def __call__(self, tensor):
         permuted_tensor = torch.permute(tensor, (0, 3, 1, 2)).float()
         return permuted_tensor
###############################################################