# Skill Boundary Oversampling

## Problem Statement

During policy evaluation, models trained on BEHAVIOR-1K demos often struggle at **skill transition boundaries**:

- **Arriving at a door → Opening the door**: Great performance
- **Finishing opening → Moving through doorway**: Failures occur
- **Navigating to a box → Bending to grasp**: Model hesitates or fails

**Root Cause**: Skill transitions are severely underrepresented in training data. Consider a typical episode:
- Navigation skill: 1000 frames (steady-state walking)
- Pick skill: 400 frames (steady-state grasping)
- **Transition**: 20-50 frames (stopping, reorienting, changing behavior)

With uniform sampling, the model sees ~96% steady-state behavior and only ~4% critical transitions.

## Solution: Boundary Oversampling

The implementation adds two new parameters to `BehaviorLeRobotDataset`:

### Parameters

#### `boundary_oversampling_factor` (float, default=1.0)
- Multiplicative factor for repeating chunks containing skill boundaries
- **Example**: `3.0` means boundary chunks appear 3× as often in training
- **Recommended values**: 2.0-5.0 for typical tasks
- Set to `1.0` to disable (baseline behavior)

#### `boundary_window_frames` (int, default=50)
- Number of frames around each boundary to mark as "boundary region"
- **Example**: `50` means frames within ±50 of a transition are marked
- Chunks overlapping these regions are oversampled
- **Recommended values**: 25-100 frames depending on transition complexity

## Implementation Details

### 1. Boundary Detection (`_build_boundary_frame_indicator`)

For each episode:
1. Load skill annotations from `annotations/task-XXXX/episode_XXXXXXXX.json`
2. Identify skill transition points (where one skill ends, next begins)
3. Handle both simple (`[start, end]`) and complex (`[[s1,e1], [s2,e2], ...]`) frame durations
   - For complex durations with multiple segments: **each segment end is treated as a boundary**
   - This captures discontinuous skills or skills with multiple phases
4. Mark frames within `±boundary_window_frames` of each transition
5. Build a global boolean array: `boundary_frame_indicator[i]` = True if frame i is near a boundary

### 2. Chunk Oversampling (`_get_keyframe_chunk_indices`)

When building chunks:
1. For each 250-frame chunk, check if it contains any boundary frames
2. If yes: Add the chunk `boundary_oversampling_factor` times **and skip filtering** (boundaries always override)
3. If no: Apply normal filtering logic (check `valid_frame_mask`), add once if valid
4. Result: Boundary chunks are ALWAYS included (even if marked invalid by filtering) and appear multiple times

**Key Design Decision**: Boundaries are so critical that they override skill-based filtering. Even if a chunk would normally be filtered out (e.g., due to undersampling "move to"), if it contains a skill boundary, it's kept and oversampled.

### 3. Integration with Existing Features

The boundary oversampling **composes cleanly** with skill undersampling:

```python
dataset = BehaviorLeRobotDataset(
    # Undersample overrepresented skills
    undersampled_skill_descriptions={
        "move to": 0.3,  # Keep 30% of navigation
    },
    # Oversample underrepresented boundaries
    boundary_oversampling_factor=4.0,
    boundary_window_frames=50,
)
```

This creates a **balanced curriculum**:
- Reduces redundant steady-state navigation
- Increases critical transition moments
- Model sees diverse behaviors with proper emphasis

## Usage Examples

### Basic Boundary Oversampling

```python
from OmniGibson.omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset

dataset = BehaviorLeRobotDataset(
    repo_id="behavior/demo-dataset",
    root="/path/to/data",
    tasks=["moving boxes to storage"],
    
    # Enable boundary oversampling
    boundary_oversampling_factor=3.0,
    boundary_window_frames=50,
    
    chunk_streaming_using_keyframe=True,
    shuffle=True,
)
```

### Combined with Skill Balancing

```python
dataset = BehaviorLeRobotDataset(
    repo_id="behavior/demo-dataset",
    root="/path/to/data",
    tasks=["setting table", "cleaning room"],
    
    # Balance skill distribution
    undersampled_skill_descriptions={
        "move to": 0.25,      # Navigation is too common
        "open door": 1.0,     # Keep all door interactions
        "pick up from": 1.0,  # Keep all grasping
    },
    
    # Emphasize boundaries
    boundary_oversampling_factor=5.0,
    boundary_window_frames=75,
    
    chunk_streaming_using_keyframe=True,
    shuffle=True,
    seed=42,
)
```

## Recommendations by Task Type

### Navigation-Heavy Tasks (e.g., "moving boxes to storage")
- `boundary_oversampling_factor=4.0-5.0` (transitions are rare)
- `boundary_window_frames=50-75` (transitions involve stopping, reorienting)
- Undersample "move to" skill aggressively (0.2-0.3)

### Manipulation-Heavy Tasks (e.g., "setting table")
- `boundary_oversampling_factor=2.0-3.0` (more balanced)
- `boundary_window_frames=25-50` (faster transitions)
- Focus on pick/place boundaries

### Complex Multi-Step Tasks (e.g., "preparing meal")
- `boundary_oversampling_factor=3.0-4.0`
- `boundary_window_frames=50-100` (complex state changes)
- Undersample repetitive sub-skills

## Expected Results

### Training Distribution Shift

**Before (Uniform Sampling)**:
- 70% Navigation steady-state
- 20% Manipulation steady-state
- 10% Transitions + Other

**After (boundary_oversampling_factor=4.0)**:
- 40% Navigation steady-state
- 20% Manipulation steady-state
- 40% Transitions (4× increase!)

### Evaluation Improvements

Expected improvements in policy rollouts:
- ✓ **Smoother transitions** between skills
- ✓ **Better initiation** of manipulation (bending, approaching)
- ✓ **Improved door traversal** after opening
- ✓ **More reliable** pick/place sequencing
- ✓ **Reduced hesitation** at skill boundaries

## Monitoring and Tuning

### Log Messages

When initializing the dataset, you'll see:

```
Building boundary frame indicator with window=50 frames, oversampling factor=3.0
Boundary frame indicator built: 45000/500000 frames marked as boundary (9.0%), 180 boundaries detected
Boundary oversampling: 60 unique chunks contain boundaries and are repeated 3x (total: 180 instances)
```

### Tuning Guidelines

1. **Start conservative**: `boundary_oversampling_factor=2.0`
2. **Monitor training loss**: Boundary frames should have improving loss
3. **Evaluate rollouts**: Check transition success rates
4. **Increase gradually**: Try 3.0, 4.0, 5.0 based on results
5. **Adjust window**: If transitions are very quick/long, change `boundary_window_frames`

### Warning Signs

⚠️ **Oversampling too much** (e.g., factor=10.0):
- Model overfits to transition patterns
- Poor performance on steady-state behavior
- Training loss plateaus or increases

⚠️ **Window too small** (e.g., 10 frames):
- Misses important transition context
- Chunks may not capture full transition

⚠️ **Window too large** (e.g., 200 frames):
- Marks too much data as "boundary"
- Dilutes the focused oversampling effect

## Technical Notes

### Reproducibility
- Uses `seed` parameter for consistent RNG
- Same seed → same boundaries marked → deterministic oversampling
- Important for experiment reproducibility

### Memory Efficiency
- Boundary indicator is a boolean array (1 bit per frame)
- Overhead: ~1 MB per 8 million frames
- Chunk duplication is pointer-based (no data duplication)

### Compatibility
- Works with chunk streaming mode (required for seg_instance_id)
- Compatible with existing skill undersampling
- Respects valid_frame_mask filtering
- Works with multi-worker DataLoader

## Related Techniques

This implementation is inspired by similar approaches in:
- **Temporal Action Detection**: Hard negative mining at boundaries
- **Action Segmentation**: Weighted loss for transition frames
- **Imitation Learning**: Demonstration augmentation near trajectory changes

Key difference: Our approach operates at the **chunk level** (compatible with video GOP structure) rather than frame level.

## Future Extensions

Potential enhancements:
1. **Adaptive oversampling**: Higher factor for rare transitions
2. **Asymmetric windows**: Different sizes before/after boundary
3. **Skill-specific factors**: More oversampling for complex transitions
4. **Boundary difficulty weighting**: Use evaluation metrics to identify hard transitions

## Citation

If this technique helps your research, please cite the BEHAVIOR-1K benchmark paper and mention the boundary oversampling feature.

