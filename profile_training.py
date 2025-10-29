

def main(compile_model, batch_size):
	# Configuration
	BATCH_SIZE = batch_size
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
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
	
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
	throughput = throughput_benchmark(model, train_loader, device, num_batches=100)
	
	# 3. Training profiling
	train_timings = profile_training(model, train_loader, optimizer, criterion, device, num_batches=50)
	
	# 4. Inference profiling
	inference_timings = profile_inference(model, train_loader, device, num_batches=50)
	
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
	parser = argparse.ArgumentParser(description="Profile Training Performance")
	parser.add_argument("compile_model", type=bool, default=False, help="JIT Compilation of Compute Graph using torch.compile")
	parser.add_argument("num_workers", type=int, default=0, help="Set num_workers in PyTorch DataLoader")
	parser.add_argument("batch_size", type=int, default=32, help="Batch size")

	args = parser.parse_args()

	print(f"Compile Model: {args.compile_model}")
	print(f"Batch Size: {args.batch_size}")
	
	main(args.compile_model, args.batch_size)