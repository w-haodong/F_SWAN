"""Tensor operations shared by the training loss and the prediction decoder."""


def gather_spatial_features(features, indices):
    """Gather B,C,H,W maps at B,K flattened locations without a dense transpose."""
    expanded = indices.unsqueeze(1).expand(-1, features.size(1), -1)
    return features.flatten(2).gather(2, expanded).transpose(1, 2).contiguous()
