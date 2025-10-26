import torch.nn as nn
import torch.nn.functional as F

"""
A Transformer implementation using only PyTorch "primitives" like nn.Linear
"""
class MLP(nn.Module):

	def __init__(self, in_dim, h_dim, out_dim, activation) -> None:
		super().__init__()
		self.activation = activation
		self.fc0 = nn.Linear(in_dim, h_dim, bias=True)
		self.fc1 = nn.Linear(h_dim, out_dim, bias=True)

	def forward(self, x):
		x = self.activation(self.fc0(x))
		return self.fc1(x)

class Attention(nn.Module):

	def __init__(self, ):
		super().__init__()


class TransformerLayer(nn.Module):

	def __init__(self, ):
		super().__init__()
