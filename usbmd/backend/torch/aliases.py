"""Convert Torch ops to numpy ops syntax used in main ops module."""

import torch

torch.expand_dims = torch.unsqueeze
torch.convert_to_tensor = torch.tensor
torch.shape = lambda x: x.shape
torch.take_along_axis = torch.take_along_dim
torch.cast = lambda x, dtype: x.type(dtype)
torch.concatenate = torch.cat
torch.iscomplex = torch.is_complex
