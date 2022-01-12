import torch


def expand_as_right(tensor: torch.Tensor, dest: torch.Tensor):
    """Expand a tensor on the right to match another tensor shape.
        Args:
            tensor: tensor to be expanded
            dest: tensor providing the target shape

        Returns: a tensor with shape matching the dest input tensor shape.

        Examples:
            >>> tensor = torch.zeros(3,4)
            >>> dest = torch.zeros(3,4,5)
            >>> print(expand_as_right(tensor, dest).shape)  # returns torch.SIze([3,4,5])
    """

    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less dimensions than the input tensor, got"
            f"got tensor.ndimension()={tensor.ndimension()} and dest.ndimension()={dest.ndimension()}")
    if not (
            tensor.shape == dest.shape[:tensor.ndimension()]
    ):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, got: tensor.shape={tensor.shape}, dest={dest.shape}")
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand_as(dest)
