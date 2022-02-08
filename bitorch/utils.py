import torch


def flatten_2D(tensor: torch.tensor) -> torch.tensor:
    """Flatten batch dimensions.

    :param tensor: tensor of any dimension (can be None)
    :return: tensor with batch dimensions flatten

    Example:

        >>> t = torch.rand((2, 3, 4))
        >>> t2 = flatten_2D(t)
        >>> t2.shape
        torch.Size([6, 4])
        >>> torch.all(t2.view((2, 3, 4)) == t)
        tensor(True)
        >>> flatten_2D(None) is None
        True
    """
    if tensor is None or len(tensor.shape) == 2:
        tensor_flatten = tensor
    elif len(tensor.shape) > 2:
        tensor_flatten = torch.flatten(tensor, end_dim=-2)
    else:
        tensor_flatten = tensor.unsqueeze(0)
    return tensor_flatten
