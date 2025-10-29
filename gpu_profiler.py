"""
GPU Profiling Script for DNA Transformer
Profiles training and inference performance with CUDA events and PyTorch profiler
"""

import torch, time, json, argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn

# Import your model (assuming it's in dna_transformer.py)
from transformer_v2 import DNATransformer, DNATokenizer, DNADataset


class PerformanceProfiler:
	"""Simple profiler using CUDA events for accurate GPU timing"""
	
	def __init__(self):
		self.timings = {}
		
	def start(self, name):
		"""Start timing a section"""
		if torch.cuda.is_available():
			start = torch.cuda.Event(enable_timing=True)
			end = torch.cuda.Event(enable_timing=True)
			start.record()
			return (name, start, end)
		else:
			return (name, time.time(), None)
	
	def end(self, ctx):
		"""End timing a section"""
		name, start, end_event = ctx
		if torch.cuda.is_available():
			end_event.record()
			torch.cuda.synchronize()
			elapsed = start.elapsed_time(end_event)  # milliseconds
		else:
			elapsed = (time.time() - start) * 1000  # convert to ms
		
		if name not in self.timings:
			self.timings[name] = []
		self.timings[name].append(elapsed)
	
	def summary(self):
		"""Print summary statistics"""
		print("\n" + "="*80)
		print("PERFORMANCE SUMMARY")
		print("="*80)
		for name, times in self.timings.items():
			mean_time = sum(times) / len(times)
			min_time = min(times)
			max_time = max(times)
			print(f"\n{name}:")
			print(f"  Mean: {mean_time:.2f} ms")
			print(f"  Min:  {min_time:.2f} ms")
			print(f"  Max:  {max_time:.2f} ms")
			print(f"  Calls: {len(times)}")
		print("="*80)
		
		return self.timings


def profile_training(model, dataloader, optimizer, criterion, device, async_dma, num_batches=50):
	"""Profile training performance"""
	print("\nProfiling Training...")
	
	profiler = PerformanceProfiler()
	model.train()
	
	for batch_idx, batch in enumerate(dataloader):
		if batch_idx >= num_batches:
			break
		
		# Data transfer
		ctx = profiler.start("data_transfer")
		input_ids = batch['input_ids'].to(device, non_blocking=async_dma)
		labels = batch['label'].to(device, non_blocking=async_dma)
		profiler.end(ctx)
		
		optimizer.zero_grad()
		
		# Forward pass
		ctx = profiler.start("forward")
		logits = model(input_ids)
		profiler.end(ctx)
		
		# Loss computation
		ctx = profiler.start("loss")
		loss = criterion(logits, labels)
		profiler.end(ctx)
		
		# Backward pass
		ctx = profiler.start("backward")
		loss.backward()
		profiler.end(ctx)
		
		# Optimizer step
		ctx = profiler.start("optimizer")
		optimizer.step()
		profiler.end(ctx)
		
		if batch_idx % 10 == 0:
			print(f"  Batch {batch_idx}/{num_batches}")
	
	return profiler.summary()


def profile_inference(model, dataloader, device, num_batches=50):
	"""Profile inference performance"""
	print("\nProfiling Inference...")
	
	profiler = PerformanceProfiler()
	model.eval()
	
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):
			if batch_idx >= num_batches:
				break
			
			# Data transfer
			ctx = profiler.start("inference_data_transfer")
			input_ids = batch['input_ids'].to(device)
			profiler.end(ctx)
			
			# Inference
			ctx = profiler.start("inference_forward")
			logits = model(input_ids)
			profiler.end(ctx)
			
			if batch_idx % 10 == 0:
				print(f"  Batch {batch_idx}/{num_batches}")
	
	return profiler.summary()


def profile_with_pytorch_profiler(model, dataloader, device, num_batches=20):
	"""Profile using PyTorch's built-in profiler (generates Chrome trace)"""
	print("\nRunning PyTorch Profiler (Chrome trace)...")
	
	from torch.profiler import profile, record_function, ProfilerActivity
	
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	criterion = nn.CrossEntropyLoss()
	
	activities = [ProfilerActivity.CPU]
	if torch.cuda.is_available():
		activities.append(ProfilerActivity.CUDA)
	
	with profile(
		activities=activities,
		record_shapes=True,
		profile_memory=True,
		with_stack=True
	) as prof:
		for batch_idx, batch in enumerate(dataloader):
			if batch_idx >= num_batches:
				break
			
			input_ids = batch['input_ids'].to(device)
			labels = batch['label'].to(device)
			
			optimizer.zero_grad()
			
			with record_function("forward"):
				logits = model(input_ids)
			
			with record_function("loss"):
				loss = criterion(logits, labels)
			
			with record_function("backward"):
				loss.backward()
			
			with record_function("optimizer_step"):
				optimizer.step()
	
	# Save trace
	trace_path = "profiler_trace.json"
	prof.export_chrome_trace(trace_path)
	print(f"Saved Chrome trace to: {trace_path}")
	print(f"   Open in Chrome at: chrome://tracing")
	
	# Print summary
	print("\n" + "="*80)
	print("TOP CUDA OPERATIONS (by time)")
	print("="*80)
	print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
	
	print("\n" + "="*80)
	print("TOP CPU OPERATIONS (by time)")
	print("="*80)
	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
	
	return prof


def memory_profiling(model, dataloader, device, batch_size=32):
	"""Profile GPU memory usage"""
	print("\nProfiling GPU Memory...")
	
	if not torch.cuda.is_available():
		print("  Skipping: CUDA not available")
		return
	
	torch.cuda.reset_peak_memory_stats()
	torch.cuda.empty_cache()
	
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	criterion = nn.CrossEntropyLoss()
	
	# Get one batch
	batch = next(iter(dataloader))
	input_ids = batch['input_ids'].to(device)
	labels = batch['label'].to(device)
	
	# Measure memory before forward
	torch.cuda.synchronize()
	mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
	
	# Forward
	logits = model(input_ids)
	loss = criterion(logits, labels)
	torch.cuda.synchronize()
	mem_after_forward = torch.cuda.memory_allocated() / 1024**2
	
	# Backward
	loss.backward()
	torch.cuda.synchronize()
	mem_after_backward = torch.cuda.memory_allocated() / 1024**2
	
	# Peak memory
	mem_peak = torch.cuda.max_memory_allocated() / 1024**2
	
	print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"  Batch size: {batch_size}")
	print(f"  Sequence length: {input_ids.shape[1]}")
	print(f"\n  Memory usage:")
	print(f"    Base (model weights):     {mem_before:.2f} MB")
	print(f"    After forward pass:       {mem_after_forward:.2f} MB  (+{mem_after_forward-mem_before:.2f} MB)")
	print(f"    After backward pass:      {mem_after_backward:.2f} MB  (+{mem_after_backward-mem_after_forward:.2f} MB)")
	print(f"    Peak memory:              {mem_peak:.2f} MB")
	print(f"\n  Memory breakdown:")
	print(f"    Activations (forward):    ~{mem_after_forward-mem_before:.2f} MB")
	print(f"    Gradients (backward):     ~{mem_after_backward-mem_after_forward:.2f} MB")


def throughput_benchmark(model, dataloader, device, async_dma, num_batches=100):
	"""Measure throughput (samples/sec)"""
	print("\nMeasuring Throughput...")
	
	model.eval()
	
	# Warmup
	print("  Warming up...")
	with torch.no_grad():
		for i, batch in enumerate(dataloader):
			if i >= 10:
				break
			input_ids = batch['input_ids'].to(device)
			_ = model(input_ids)
	
	# Actual benchmark
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	
	start_time = time.time()
	total_samples = 0
	
	with torch.no_grad():
		for batch_idx, batch in enumerate(dataloader):
			if batch_idx >= num_batches:
				break
			input_ids = batch['input_ids'].to(device, non_blocking=async_dma)
			_ = model(input_ids)
			total_samples += input_ids.shape[0]
	
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	
	elapsed = time.time() - start_time
	throughput = total_samples / elapsed
	
	print(f"\n  Total samples: {total_samples}")
	print(f"  Time elapsed: {elapsed:.2f} seconds")
	print(f"  Throughput: {throughput:.2f} samples/sec")
	print(f"  Time per sample: {1000/throughput:.2f} ms")
	
	return throughput


def main(compile_model, num_workers, async_dma, batch_size):
	# Configuration
	MAX_LENGTH = 512
	D_MODEL = 128
	NHEAD = 8
	NUM_LAYERS = 4
	DIM_FEEDFORWARD = 512
	DROPOUT = 0.1
	
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Profiling on: {device}")
	if torch.cuda.is_available():
		print(f"   GPU: {torch.cuda.get_device_name(0)}")
		print(f"   CUDA Version: {torch.version.cuda}")
		print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
	
	# Load dataset
	print("\nLoading dataset...")
	train_data = HumanEnhancersCohn(split='train', version=0)
	tokenizer = DNATokenizer()
	
	# Create smaller dataset for profiling
	train_dataset = DNADataset(train_data, tokenizer, max_length=MAX_LENGTH)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=async_dma)
	
	# Initialize model
	print("\nInitializing model...")
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

	if compile_model:
		model = torch.compile(model)
	
	print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
	
	# Setup training components
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	# Run profiling
	print("\n" + "="*80)
	print("STARTING PROFILING SUITE")
	print("="*80)
	
	# 1. Memory profiling
	# memory_profiling(model, train_loader, device, BATCH_SIZE)
	
	# 2. Throughput benchmark
	throughput = throughput_benchmark(model, train_loader, device, async_dma, num_batches=200)
	
	# 3. Training profiling
	train_timings = profile_training(model, train_loader, optimizer, criterion, device, async_dma, num_batches=200)
	
	# 4. Inference profiling
	# inference_timings = profile_inference(model, train_loader, device, num_batches=50)
	
	# 5. PyTorch profiler (Chrome trace)
	# pytorch_prof = profile_with_pytorch_profiler(model, train_loader, device, num_batches=20)
	
	# Save results
	results = {
		'device': str(device),
		'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
		'batch_size': BATCH_SIZE,
		'max_length': MAX_LENGTH,
		'd_model': D_MODEL,
		'num_layers': NUM_LAYERS,
		'num_params': sum(p.numel() for p in model.parameters()),
		'throughput_samples_per_sec': throughput,
		'training_timings': {k: sum(v)/len(v) for k, v in train_timings.items()},
		'inference_timings': {k: sum(v)/len(v) for k, v in inference_timings.items()},
	}
	
	with open('profiling_results.json', 'w') as f:
		json.dump(results, f, indent=2)
	
	print("\nProfiling complete! Results saved to profiling_results.json")
	print("   Chrome trace saved to profiler_trace.json")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Profile Model Implementations")
	parser.add_argument("--compile_model", action="store_true", help="Compile Model")
	parser.add_argument("--no_compile_model", dest="compile_model", action="store_false")
	parser.add_argument("--async_dma", action="store_true", help="Enable async DMA")
	parser.add_argument("--no_async_dma", dest="async_dma", action="store_false")
	parser.add_argument("num_workers", type=int, default=0, help="Set num_workers in PyTorch DataLoader")
	parser.add_argument("batch_size", type=int, default=32, help="Batch size")

	args = parser.parse_args()

	print(f"Compile Model: {args.compile_model}")
	print(f"num_workers: {args.num_workers}")
	print(f"async_dma: {args.async_dma}")
	print(f"Batch Size: {args.batch_size}")

	main(args.compile_model, args.num_workers, args.async_dma, args.batch_size)




