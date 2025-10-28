"""
The same transformer model written using PyTorch transformer modules
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn


# ==================== Data Preparation ====================

class DNATokenizer:
	"""Simple tokenizer for DNA sequences (A, C, G, T)"""
	def __init__(self):
		self.vocab = {'<PAD>': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '<UNK>': 5}
		self.vocab_size = len(self.vocab)
	
	def encode(self, sequence):
		"""Convert DNA string to list of token IDs"""
		return [self.vocab.get(char, self.vocab['<UNK>']) for char in sequence.upper()]


class DNADataset(Dataset):
	"""PyTorch Dataset wrapper for genomic sequences"""
	def __init__(self, hf_dataset, tokenizer, max_length=512):
		self.data = hf_dataset
		self.tokenizer = tokenizer
		self.max_length = max_length
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		item = self.data[idx]
		sequence = item[0]
		label = item[1]
		
		# Tokenize sequence
		tokens = self.tokenizer.encode(sequence)
		
		# Truncate or pad
		if len(tokens) > self.max_length:
			tokens = tokens[:self.max_length]
		else:
			tokens = tokens + [self.tokenizer.vocab['<PAD>']] * (self.max_length - len(tokens))
		
		return {
			'input_ids': torch.tensor(tokens, dtype=torch.long),
			'label': torch.tensor(label, dtype=torch.long)
		}


# ==================== Transformer Components ====================

class PositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding"""
	def __init__(self, d_model, max_len=5000, dropout=0.1):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Create positional encoding matrix
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # Add batch dimension
		
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		# x shape: (batch_size, seq_len, d_model)
		x = x + self.pe[:, :x.size(1), :]
		return self.dropout(x)


class DNATransformer(nn.Module):
	"""Transformer model for DNA sequence classification"""
	def __init__(
		self,
		vocab_size,
		d_model=128,
		nhead=8,
		num_layers=4,
		dim_feedforward=512,
		max_seq_length=512,
		num_classes=2,
		dropout=0.1,
	):
		super().__init__()
		
		self.d_model = d_model
		
		# Token embedding
		self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
		
		# Positional encoding
		self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

		# # Learned CLS token
		# cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
		# self.register_buffer('cls_token', cls_token)
		
		# Transformer encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			norm_first=True,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		
		# Classification head
		self.classifier = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model, num_classes)
		)
		
		self._init_weights()
	
	def _init_weights(self):
		"""Initialize weights"""
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
	
	def forward(self, input_ids):
		# input_ids shape: (batch_size, seq_len)
		
		# Create padding mask
		padding_mask = (input_ids == 0)  # True for padding tokens

		# cls_pad = torch.zeros((input_ids.size(0), 1), dtype=torch.bool)
		# self.register_buffer('cls_pad', cls_pad)
		# padding_mask = torch.cat([cls_pad, padding_mask], dim=1)
		
		# Embed tokens
		x = self.embedding(input_ids) * math.sqrt(self.d_model)
		
		# Add positional encoding
		x = self.pos_encoder(x)

		# # Add CLS token at start of sequence
		# cls_tokens = self.cls_token.expand(input_ids.size(0), -1, -1)  # [B, 1, D]
		# x = torch.cat((cls_tokens, x), dim=1)  # [B, L+1, D]
		
		# Pass through transformer 
		x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
		
		# Global average pooling (excluding padding)
		mask = (~padding_mask).unsqueeze(-1).float()
		x = (x * mask).sum(dim=1) / mask.sum(dim=1)

		# x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
		# x = x.max(dim=1).values

		# # CLS representation
		# cls_repr = x[:, 0, :]  # [B, D]
		
		# Classification
		logits = self.classifier(x)
		
		return logits


# ==================== Training Functions ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
	"""Train for one epoch"""
	model.train()
	total_loss = 0
	correct = 0
	total = 0
	
	pbar = tqdm(dataloader, desc='Training')
	for batch in pbar:
		input_ids = batch['input_ids'].to(device)
		labels = batch['label'].to(device)
		
		optimizer.zero_grad()
		
		# Forward pass
		logits = model(input_ids)
		loss = criterion(logits, labels)
		
		# Backward pass
		loss.backward()
		optimizer.step()
		
		# Statistics
		total_loss += loss.item()
		_, predicted = torch.max(logits, 1)
		correct += (predicted == labels).sum().item()
		total += labels.size(0)
		
		pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
	
	return total_loss / len(dataloader), 100 * correct / total


def evaluate(model, dataloader, criterion, device):
	"""Evaluate the model"""
	model.eval()
	total_loss = 0
	correct = 0
	total = 0
	
	with torch.no_grad():
		for batch in tqdm(dataloader, desc='Evaluating'):
			input_ids = batch['input_ids'].to(device)
			labels = batch['label'].to(device)
			
			logits = model(input_ids)
			loss = criterion(logits, labels)
			
			total_loss += loss.item()
			_, predicted = torch.max(logits, 1)
			correct += (predicted == labels).sum().item()
			total += labels.size(0)
	
	return total_loss / len(dataloader), 100 * correct / total


# ==================== Main Training Script ====================

def main():
	# Hyperparameters
	BATCH_SIZE = 32
	MAX_LENGTH = 512
	D_MODEL = 128
	NHEAD = 8
	NUM_LAYERS = 4
	DIM_FEEDFORWARD = 512
	DROPOUT = 0.1
	LEARNING_RATE = 1e-3
	NUM_EPOCHS = 10
	
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	if torch.cuda.is_available():
		print(f"GPU: {torch.cuda.get_device_name(0)}")
		print(f"CUDA Version: {torch.version.cuda}")
	
	# Load dataset
	print("Loading dataset...")
	train_data = HumanEnhancersCohn(split='train', version=0)
	test_data = HumanEnhancersCohn(split='test', version=0)
	
	# Initialize tokenizer
	tokenizer = DNATokenizer()

	# Create PyTorch datasets
	train_dataset = DNADataset(train_data, tokenizer, max_length=MAX_LENGTH)
	test_dataset = DNADataset(test_data, tokenizer, max_length=MAX_LENGTH)

	train_dataset_onebatch = [train_dataset[i] for i in range(BATCH_SIZE*4)]
	test_dataset_onebatch = [test_dataset[i] for i in range(BATCH_SIZE)]

	# Create dataloaders
	train_loader = DataLoader(train_dataset_onebatch, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
	
	print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
	
	# Initialize model
	model = DNATransformer(
		vocab_size=tokenizer.vocab_size,
		d_model=D_MODEL,
		nhead=NHEAD,
		num_layers=NUM_LAYERS,
		dim_feedforward=DIM_FEEDFORWARD,
		max_seq_length=MAX_LENGTH,
		num_classes=2,
		dropout=DROPOUT
	).to(device)

	# model = torch.compile(model)
	
	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	
	# Training loop
	best_test_acc = 0
	for epoch in range(NUM_EPOCHS):
		print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
		
		train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)
		
		print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
		print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
		
		# Save best model
		if test_acc > best_test_acc:
			best_test_acc = test_acc
			torch.save(model.state_dict(), 'best_model.pt')
			print(f"Saved best model with test acc: {test_acc:.2f}%")
	
	print(f"\nBest test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
	main()