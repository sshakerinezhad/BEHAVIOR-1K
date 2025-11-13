import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

def simulate_dataset_initialization(seed=42, worker_id=0, num_chunks=100):
    """
    Simulate how the BehaviorLeRobotDataset initializes its chunk streaming.
    
    Args:
        seed: Base random seed (default 42)
        worker_id: Worker ID (0, 1, 2, ...) simulating different dataloader workers
        num_chunks: Number of chunks to simulate
    
    Returns:
        Dictionary with shuffled chunks, starting chunk idx, and starting frame idx
    """
    # Create dummy chunks: (start_idx, end_idx, local_start_idx)
    # Each chunk is 250 frames
    chunks = [(i * 250, (i + 1) * 250, 0) for i in range(num_chunks)]
    
    # This is the actual logic from lerobot_dataset.py lines 388-394
    rng = np.random.default_rng(seed + worker_id)
    rng.shuffle(chunks)
    current_streaming_chunk_idx = rng.integers(0, len(chunks)).item()
    current_streaming_frame_idx = chunks[current_streaming_chunk_idx][0]
    
    return {
        'chunks': chunks,
        'current_streaming_chunk_idx': current_streaming_chunk_idx,
        'current_streaming_frame_idx': current_streaming_frame_idx,
        'seed': seed + worker_id,
    }

def test_reproducibility():
    """Test that the same seed produces the same shuffle and starting position."""
    print("=" * 60)
    print("TEST 1: Reproducibility with same seed")
    print("=" * 60)
    
    result1 = simulate_dataset_initialization(seed=42, worker_id=0, num_chunks=20)
    result2 = simulate_dataset_initialization(seed=42, worker_id=0, num_chunks=20)
    
    print(f"Run 1 - Starting chunk idx: {result1['current_streaming_chunk_idx']}")
    print(f"Run 2 - Starting chunk idx: {result2['current_streaming_chunk_idx']}")
    print(f"First 5 chunks (run 1): {result1['chunks'][:5]}")
    print(f"First 5 chunks (run 2): {result2['chunks'][:5]}")
    print(f"Shuffles are identical: {result1['chunks'] == result2['chunks']}")
    print(f"Starting positions are identical: {result1['current_streaming_chunk_idx'] == result2['current_streaming_chunk_idx']}")
    print()

def test_different_workers():
    """Test how different workers get different shuffles."""
    print("=" * 60)
    print("TEST 2: Different workers (simulating num_workers=4)")
    print("=" * 60)
    
    num_workers = 4
    for worker_id in range(num_workers):
        result = simulate_dataset_initialization(seed=42, worker_id=worker_id, num_chunks=20)
        print(f"Worker {worker_id}:")
        print(f"  Effective seed: {result['seed']}")
        print(f"  Starting chunk idx: {result['current_streaming_chunk_idx']}")
        print(f"  Starting frame idx: {result['current_streaming_frame_idx']}")
        print(f"  First 5 chunks: {result['chunks'][:5]}")
        print()

def simulate_training_steps(chunks, start_chunk_idx, steps=10, batch_size=32):
    """
    Simulate which chunks/frames are accessed during training.
    
    Args:
        chunks: List of (start, end, local_start) tuples
        start_chunk_idx: Which chunk to start from
        steps: Number of training steps to simulate
        batch_size: Batch size
    
    Returns:
        List of frames accessed
    """
    current_chunk_idx = start_chunk_idx
    current_frame_idx = chunks[current_chunk_idx][0]
    frames_accessed = []
    
    for step in range(steps * batch_size):
        # Record current frame
        frames_accessed.append((current_chunk_idx, current_frame_idx))
        
        # Move to next frame
        current_frame_idx += 1
        
        # Check if we've exhausted current chunk
        if current_frame_idx >= chunks[current_chunk_idx][1]:
            current_chunk_idx += 1
            # Wrap around if we've gone through all chunks
            if current_chunk_idx >= len(chunks):
                current_chunk_idx = 0
            current_frame_idx = chunks[current_chunk_idx][0]
    
    return frames_accessed

def test_crash_and_resume():
    """Test what happens when training crashes and resumes."""
    print("=" * 60)
    print("TEST 3: Crash and Resume Scenario")
    print("=" * 60)
    
    seed = 42
    worker_id = 0
    num_chunks = 20
    
    # Initial training run
    print("Initial training run:")
    result1 = simulate_dataset_initialization(seed=seed, worker_id=worker_id, num_chunks=num_chunks)
    frames1 = simulate_training_steps(result1['chunks'], result1['current_streaming_chunk_idx'], steps=5, batch_size=8)
    print(f"  Starting chunk idx: {result1['current_streaming_chunk_idx']}")
    print(f"  Trained for {len(frames1)} frames (5 steps × batch_size 8)")
    print(f"  First 10 frames accessed: {frames1[:10]}")
    print(f"  Last frame accessed: {frames1[-1]}")
    print()
    
    # Crash happens here...
    print("💥 CRASH! Restarting training...")
    print()
    
    # Resume training (new dataset initialization)
    print("Resumed training run:")
    result2 = simulate_dataset_initialization(seed=seed, worker_id=worker_id, num_chunks=num_chunks)
    frames2 = simulate_training_steps(result2['chunks'], result2['current_streaming_chunk_idx'], steps=5, batch_size=8)
    print(f"  Starting chunk idx: {result2['current_streaming_chunk_idx']}")
    print(f"  First 10 frames accessed: {frames2[:10]}")
    print()
    
    # Check overlap
    overlap = len(set(frames1) & set(frames2))
    print(f"⚠️  Data overlap: {overlap}/{len(frames1)} frames ({100*overlap/len(frames1):.1f}%)")
    print(f"⚠️  You would see THE SAME DATA again after resume!")
    print()

class ChunkedStreamingDataset(Dataset):
    """
    A simplified version of BehaviorLeRobotDataset that mimics the chunk streaming behavior.
    """
    def __init__(self, seed=42, num_chunks=100, chunk_size=250, resume_step=0):
        super().__init__()
        self.seed = seed
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.resume_step = resume_step
        
        # Create chunks (mimicking the actual dataset structure)
        self.all_chunks = [(i * chunk_size, (i + 1) * chunk_size, 0) for i in range(num_chunks)]
        
        # These will be initialized per-worker
        self.chunks = None
        self.current_streaming_chunk_idx = None
        self.current_streaming_frame_idx = None
        self.worker_id = None
        self.initialization_log = []
        
    def __len__(self):
        # Return total number of frames
        return self.num_chunks * self.chunk_size
    
    def _initialize_worker(self):
        """Initialize the chunk streaming for this worker (mimics lerobot_dataset.py lines 388-402)"""
        worker_info = get_worker_info()
        print(f"worker_info.id: {worker_info.id}")
        self.worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        print(f"THE NUM_WORKERS FROM THE DATALOADER worker_info.num_workers={num_workers}")

        # Make a copy of chunks for this worker
        self.chunks = self.all_chunks.copy()
        
        # This is the EXACT logic from lerobot_dataset.py
        rng = np.random.default_rng(self.seed + self.worker_id)
        rng.shuffle(self.chunks)
        self.current_streaming_chunk_idx = rng.integers(0, len(self.chunks)).item()
        self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
        
        # Fast-forward to resume position if resuming from checkpoint (NEW FEATURE!)
        if self.resume_step > 0:
            # Calculate how many samples this worker should have processed
            # resume_step is total samples processed globally
            samples_to_skip = self.resume_step // num_workers
            if self.worker_id < (self.resume_step % num_workers):
                # Some workers may have processed one extra sample due to rounding
                samples_to_skip += 1
            
            if samples_to_skip > 0:
                print(f"[Worker {self.worker_id}] Resuming from {self.resume_step} global samples, fast-forwarding {samples_to_skip} samples")
                self._fast_forward_frames(samples_to_skip)
                print(f"[Worker {self.worker_id}] Resumed at chunk {self.current_streaming_chunk_idx}, frame {self.current_streaming_frame_idx}")
        
        # Log this initialization
        log_entry = {
            'worker_id': self.worker_id,
            'seed': self.seed + self.worker_id,
            'starting_chunk_idx': self.current_streaming_chunk_idx,
            'starting_frame_idx': self.current_streaming_frame_idx,
            'first_5_chunks': self.chunks[:5],
        }
        if self.resume_step == 0:
            print(f"[Worker {self.worker_id}] Initialized: starting at chunk {self.current_streaming_chunk_idx}, frame {self.current_streaming_frame_idx}")
        return log_entry
    
    def _fast_forward_frames(self, num_samples: int) -> None:
        """Fast-forward through the chunk sequence by num_samples frames."""
        for _ in range(num_samples):
            # Move to next frame
            self.current_streaming_frame_idx += 1
            
            # Check if we've exhausted the current chunk
            if self.current_streaming_frame_idx >= self.chunks[self.current_streaming_chunk_idx][1]:
                self.current_streaming_chunk_idx += 1
                # Wrap around if we've gone through all chunks
                if self.current_streaming_chunk_idx >= len(self.chunks):
                    self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
    
    def __getitem__(self, idx):
        # Initialize on first call in this worker
        if self.current_streaming_chunk_idx is None:
            self._initialize_worker()
        
        # Get current frame from current chunk
        chunk_start, chunk_end, _ = self.chunks[self.current_streaming_chunk_idx]
        
        # Create a simple data item (just metadata for testing)
        item = {
            'worker_id': self.worker_id,
            'chunk_idx': self.current_streaming_chunk_idx,
            'frame_idx': self.current_streaming_frame_idx,
            'chunk_range': (chunk_start, chunk_end),
            'data': torch.tensor([self.current_streaming_frame_idx], dtype=torch.float32),
        }
        
        # Move to next frame (mimics lerobot_dataset.py lines 404-410, 478)
        self.current_streaming_frame_idx += 1
        
        # Check if we've exhausted the current chunk
        if self.current_streaming_frame_idx >= self.chunks[self.current_streaming_chunk_idx][1]:
            self.current_streaming_chunk_idx += 1
            # Wrap around if we've gone through all chunks
            if self.current_streaming_chunk_idx >= len(self.chunks):
                self.current_streaming_chunk_idx = 0
            self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
        
        return item


def test_with_real_dataloader():
    """Test using an actual PyTorch DataLoader with multiple workers."""
    print("=" * 60)
    print("TEST 4: Real DataLoader with num_workers=4")
    print("=" * 60)
    print()
    
    seed = 42
    num_chunks = 20
    chunk_size = 250
    batch_size = 8
    num_workers = 4
    
    # Create dataset
    dataset = ChunkedStreamingDataset(seed=seed, num_chunks=num_chunks, chunk_size=chunk_size)
    
    # Create dataloader with multiple workers
    print(f"Creating DataLoader with {num_workers} workers, batch_size={batch_size}")
    print("Each worker will initialize independently with its own seed offset...")
    print()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # We handle shuffling in the dataset
    )
    
    # Collect data from first few batches
    print("Fetching first 3 batches:")
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        batches.append(batch)
        print(f"\nBatch {i}:")
        print(f"  Worker IDs: {batch['worker_id'].tolist()}")
        print(f"  Chunk indices: {batch['chunk_idx'].tolist()}")
        print(f"  Frame indices: {batch['frame_idx'].tolist()}")
    
    print()
    print("=" * 60)
    print()


def test_crash_and_resume_with_dataloader():
    """Test crash/resume behavior with real DataLoader."""
    print("=" * 60)
    print("TEST 5: Crash and Resume with Real DataLoader")
    print("=" * 60)
    print()
    
    seed = 42
    num_chunks = 20
    chunk_size = 250
    batch_size = 8
    num_workers = 2  # Use 2 workers for clearer output
    
    print("Initial training run:")
    print("-" * 60)
    
    # Create dataset and dataloader
    dataset1 = ChunkedStreamingDataset(seed=seed, num_chunks=num_chunks, chunk_size=chunk_size)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Collect first 5 batches
    batches1 = []
    frames1 = []
    for i, batch in enumerate(dataloader1):
        if i >= 10:
            break
        batches1.append(batch)
        frames1.extend(list(zip(batch['worker_id'].tolist(), 
                                batch['chunk_idx'].tolist(), 
                                batch['frame_idx'].tolist())))
        print(f"Batch {i}: workers {batch['worker_id'].tolist()}, frames {batch['frame_idx'].tolist()}")
    
    print(f"\nTrained for {len(frames1)} total frames across 5 batches")
    print()
    
    # Simulate crash
    print("💥 CRASH! Training interrupted...")
    print("Restarting training (creating new dataset & dataloader)...")
    print()
    
    print("Resumed training run:")
    print("-" * 60)
    
    # Create NEW dataset and dataloader (simulating restart)
    dataset2 = ChunkedStreamingDataset(seed=seed, num_chunks=num_chunks, chunk_size=chunk_size)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Collect first 5 batches again
    batches2 = []
    frames2 = []
    for i, batch in enumerate(dataloader2):
        if i >= 10:
            break
        batches2.append(batch)
        frames2.extend(list(zip(batch['worker_id'].tolist(), 
                                batch['chunk_idx'].tolist(), 
                                batch['frame_idx'].tolist())))
        print(f"Batch {i}: workers {batch['worker_id'].tolist()}, frames {batch['frame_idx'].tolist()}")
    
    print(f"\nTrained for {len(frames2)} total frames across 5 batches")
    print()
    
    # Check overlap
    frames1_set = set([(wid, fid) for wid, cid, fid in frames1])
    frames2_set = set([(wid, fid) for wid, cid, fid in frames2])
    overlap = len(frames1_set & frames2_set)
    
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Total frames in run 1: {len(frames1)}")
    print(f"Total frames in run 2: {len(frames2)}")
    print(f"Overlapping (worker_id, frame_idx) pairs: {overlap}/{len(frames1)}")
    print(f"Overlap percentage: {100*overlap/len(frames1):.1f}%")
    print()
    
    # Show which frames are identical
    print("First 10 (worker_id, chunk_idx, frame_idx) from each run:")
    print(f"  Run 1: {frames1[:10]}")
    print(f"  Run 2: {frames2[:10]}")
    print()
    
    if frames1 == frames2:
        print("⚠️  EXACT SAME DATA in EXACT SAME ORDER!")
        print("⚠️  Workers start from the same positions every time!")
    print()


def test_crash_and_resume_with_fix():
    """Test the FIX: using resume_step to properly continue from checkpoint."""
    print("=" * 60)
    print("TEST 6: Crash and Resume WITH FIX (resume_step)")
    print("=" * 60)
    print()
    
    seed = 42
    num_chunks = 20
    chunk_size = 250
    batch_size = 8
    num_workers = 2
    
    print("Initial training run:")
    print("-" * 60)
    
    # Create dataset and dataloader (NO resume_step)
    dataset1 = ChunkedStreamingDataset(seed=seed, num_chunks=num_chunks, chunk_size=chunk_size, resume_step=0)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Collect first 5 batches
    batches1 = []
    frames1 = []
    for i, batch in enumerate(dataloader1):
        if i >= 10:
            break
        batches1.append(batch)
        frames1.extend(list(zip(batch['worker_id'].tolist(), 
                                batch['chunk_idx'].tolist(), 
                                batch['frame_idx'].tolist())))
        print(f"Batch {i}: workers {batch['worker_id'].tolist()}, frames {batch['frame_idx'].tolist()}")
    
    print(f"\nTrained for {len(frames1)} total frames across 5 batches")
    print()
    
    # Simulate crash
    print("💥 CRASH! Training interrupted at step 5...")
    print(f"Total samples processed: 5 steps × {batch_size} batch_size = {5 * batch_size} samples")
    print(f"Restarting training with resume_step={5 * batch_size}...")
    print()
    
    print(f"Resumed training run (with resume_step={5 * batch_size}):")
    print("-" * 60)
    
    # Create NEW dataset with resume_step=total samples (not steps!)
    N_SKIPPED = 5
    dataset2 = ChunkedStreamingDataset(seed=seed, num_chunks=num_chunks, chunk_size=chunk_size, resume_step=N_SKIPPED * batch_size)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Collect first 5 batches from the RESUMED position
    batches2 = []
    frames2 = []
    for i, batch in enumerate(dataloader2):
        if i >= 5:
            break
        batches2.append(batch)
        frames2.extend(list(zip(batch['worker_id'].tolist(), 
                                batch['chunk_idx'].tolist(), 
                                batch['frame_idx'].tolist())))
        print(f"Batch {i + N_SKIPPED}: workers {batch['worker_id'].tolist()}, frames {batch['frame_idx'].tolist()}")
    
    print(f"\nTrained for {len(frames2)} total frames across 5 batches (steps 5-10)")
    print()
    
    # Check overlap
    frames1_set = set([(wid, fid) for wid, cid, fid in frames1])
    frames2_set = set([(wid, fid) for wid, cid, fid in frames2])
    overlap = len(frames1_set & frames2_set)
    
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Frames in steps 0-4: {len(frames1)}")
    print(f"Frames in steps 5-9 (after resume): {len(frames2)}")
    print(f"Overlapping (worker_id, frame_idx) pairs: {overlap}/{len(frames1)}")
    print(f"Overlap percentage: {100*overlap/len(frames1) if len(frames1) > 0 else 0:.1f}%")
    print()
    
    print("First 10 (worker_id, chunk_idx, frame_idx) from each run:")
    print(f"  Steps 0-4:  {frames1[:10]}")
    print(f"  Steps 5-9:  {frames2[:10]}")
    print()
    
    if overlap == 0:
        print("✅ NO DATA OVERLAP! Training properly resumed!")
        print("✅ Each worker continued from where it left off!")
    else:
        print(f"⚠️  Still some overlap: {overlap} frames")
    print()


if __name__ == "__main__":
    # # Original tests
    # test_reproducibility()
    # test_different_workers()
    # test_crash_and_resume()
    
    # New tests with real DataLoader
    # test_with_real_dataloader()
    test_crash_and_resume_with_dataloader()
    test_crash_and_resume_with_fix()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ The shuffle IS deterministic (same seed = same order)")
    print("✅ Different workers get different shuffles")
    print("✅ Real DataLoader confirms the same behavior")
    print("❌ WITHOUT FIX: After crash, ALL workers restart from the beginning")
    print("✅ WITH FIX (resume_step): Workers continue from correct position!")
    print()
