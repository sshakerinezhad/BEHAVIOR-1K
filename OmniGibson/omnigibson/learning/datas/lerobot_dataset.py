import datasets
from datetime import datetime
import json
import os
import numpy as np
import packaging.version
import torch as th
from collections import defaultdict
from collections.abc import Callable
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, CODEBASE_VERSION
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    STATS_PATH,
    TASKS_PATH,
    cast_stats_to_numpy,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    backward_compatible_episodes_stats,
    load_json,
    load_jsonlines,
    load_info,
    is_valid_version,
)
from lerobot.datasets.video_utils import get_safe_default_codec
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.lerobot_utils import hf_transform_to_torch, decode_video_frames, aggregate_stats
from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
from typing import Iterable, List, Tuple

EPISODES_PROMPT_VARIANTS_PATH = "meta/episodes_with_variants.jsonl"

logger = create_module_logger("BehaviorLeRobotDataset")

MODALITY_NAMES = {"rgb", "depth", "seg_instance_id"}

class BehaviorLeRobotDataset(LeRobotDataset):
    """
    BehaviorLeRobotDataset is a customized dataset class for loading and managing LeRobot datasets,
    with additional filtering and loading options tailored for the BEHAVIOR-1K benchmark.
    This class extends LeRobotDataset and introduces the following customizations:
        - Task-based filtering: Load only episodes corresponding to specific tasks.
        - Modality and camera selection: Load only specified modalities (e.g., "rgb", "depth", "seg_instance_id")
          and cameras (e.g., "left_wrist", "right_wrist", "head").
        - Ability to download and use additional annotation and metainfo files.
        - Local-only mode: Optionally restrict dataset usage to local files, disabling downloads.
        - Optional batch streaming using keyframe for faster access.
    These customizations allow for more efficient and targeted dataset usage in the context of B1K tasks
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        resampled_skill_descriptions: dict[str, float] | None = None,
        boundary_oversampling_factor: int = 1,
        boundary_window_frames: int = 50,
        checkpoint_dir: str | Path | None = None,
        resume_step: int = 0,
    ):
        """
        Custom args:
            episodes (List[int]): list of episodes to use PER TASK.
                NOTE: This is different from the actual episode indices in the dataset.
                Rather, this is meant to be used for train/val split, or loading a specific amount of partial data.
                If set to None, all episodes will be loaded for a given task.
            tasks (List[str]): list of task names to load. If None, all tasks will be loaded.
            modalities (List[str]): list of modality names to load. If None, all modalities will be loaded.
                must be a subset of ["rgb", "depth", "seg_instance_id"]
            cameras (List[str]): list of camera names to load. If None, all cameras will be loaded.
                must be a subset of ["left_wrist", "right_wrist", "head"]
            local_only (bool): whether to only use local data (not download from HuggingFace).
                NOTE: set this to False and force_cache_sync to True if you want to force re-syncing the local cache with the remote dataset.
                For more details, please refer to the `force_cache_sync` argument in the base class.
            check_timestamp_sync (bool): whether to check timestamp synchronization between different modalities and the state/action data.
                While it is set to True in the original LeRobotDataset and is set to True here by default, it can be set to False to skip the check for faster loading.
                This will especially save time if you are loading the complete challenge demo dataset.
            chunk_streaming_using_keyframe (bool): whether to use chunk streaming mode for loading the dataset using keyframes.
                When this is enabled, the dataset will pseudo-randomly load data in chunks based on keyframes, allowing for faster access to the data.
                NOTE: As B1K challenge demos has GOP size of 250 frames for efficient storage, it is STRONGLY recommended to set this to True if you don't need true frame-level random access.
                When this is enabled, it is recommended to set shuffle to True for better randomness in chunk selection.
                We also enforce that segmentation instance ID videos can only be loaded in chunk_streaming_using_keyframe mode for faster access.
            shuffle (bool): whether to shuffle the chunks after loading. This ONLY applies in chunk streaming mode. Recommended to be set to True for better randomness in chunk selection.
            seed (int): random seed for shuffling chunks and for probabilistic skill resampling.
            resampled_skill_descriptions (Dict[str, float]): dict mapping skill descriptions to their resampling factors.
                - 0.0 < factor < 1.0: undersample (probabilistic frame exclusion). For each skill with this factor,
                  the skill's frames will be included with the given probability.
                - factor > 1.0: oversample (chunk duplication). Chunks containing these skills will be duplicated
                  floor(factor) times. For example, factor=2.5 results in 2x duplication.
                - factor == 1.0 or not in dict: no resampling (included normally).
                If annotations cannot be read for an episode, all frames in that episode will be marked as invalid.
                This filtering is applied at initialization, respects chunk streaming mode, and uses the seed for reproducibility.
            boundary_oversampling_factor (int): multiplicative factor for oversampling chunks that contain skill boundaries.
                For example, 3 means chunks containing boundaries will appear 3x as often in training.
                Set to 1 (default) to disable boundary oversampling.
                This helps the model learn critical transition moments between skills which are often underrepresented.
            boundary_window_frames (int): number of frames around each skill boundary to consider as "boundary region".
                For example, 50 means frames within ±50 of a skill transition are marked as boundary frames.
                Chunks overlapping with these regions will be oversampled according to boundary_oversampling_factor.
            checkpoint_dir (Path | None): directory to save the chunks file for chunk streaming.
            resume_step (int): total number of samples processed globally across all workers before resuming.
                If > 0, each worker will fast-forward to the appropriate position in their shuffled chunk sequence.
                This ensures training continues from where it left off after a crash/restart.
                Note: This should be (training_step * batch_size), NOT just training_step. For example, if you've
                completed 1000 training steps with batch_size=32, pass resume_step=32000.
        """
        Dataset.__init__(self)
        self.checkpoint_dir = checkpoint_dir
        self.resume_step = resume_step
        self.repo_id = repo_id
        self.root = Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Caches for per-episode sidecar files
        self._skill_prompts_cache = {}
        # self._annotations_cache = {}

        # ========== Customizations ==========
        self.seed = seed
        resampled_skill_descriptions = resampled_skill_descriptions if resampled_skill_descriptions is not None else {}
        # Split into undersampled and oversampled
        self.undersampled_skill_descriptions = {k: v for k, v in resampled_skill_descriptions.items() if v < 1.0}
        self.oversampled_skill_descriptions = {k: int(v) for k, v in resampled_skill_descriptions.items() if v > 1.0}
        self.boundary_oversampling_factor = boundary_oversampling_factor
        self.boundary_window_frames = boundary_window_frames
        if modalities is None:
            modalities = ["rgb", "depth", "seg_instance_id"]
        if "seg_instance_id" in modalities:
            assert chunk_streaming_using_keyframe, "For the sake of data loading speed, please use chunk_streaming_using_keyframe=True when loading segmentation instance ID videos."
        if "depth" in modalities:
            assert self.video_backend == "pyav", (
                "Depth videos can only be decoded with the 'pyav' backend. "
                "Please set video_backend='pyav' when initializing the dataset."
            )
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]
        # Load metadata
        self.meta = BehaviorLerobotDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )
        # overwrite episode based on task
        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        # get the episodes grouped by task
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["episode_index"] // 1e4 in self.meta.tasks:
                epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        # sort and cherrypick episodes within each task
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [epi_by_task[task_id][i] for i in episodes if i < len(epi_by_task[task_id])]
        # now put episodes back together
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])

        # record the positional index of each episode index within self.episodes
        self.episode_data_index_pos = {ep_idx: i for i, ep_idx in enumerate(self.episodes)}
        logger.info(f"Total episodes: {len(self.episodes)}")
        # ====================================

        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), f"Missing file: {self.root / fpath}"
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        # Apply frame-level filtering based on skill annotations if needed
        # This must happen BEFORE creating chunks
        self.valid_frame_mask = None
        self.boundary_frame_indicator = None
        self.oversampled_skill_indicator = None
        if self.undersampled_skill_descriptions:
            self._build_undersampled_frame_mask()
        if self.oversampled_skill_descriptions:
            self._build_oversampled_skill_indicator()
        if self.boundary_oversampling_factor > 1:
            self._build_boundary_frame_indicator()
        
        # handle streaming mode and shuffling of episodes
        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            if not shuffle:
                logger.warning(
                    "chunk_streaming_using_keyframe mode is enabled but shuffle is set to False. This may lead to less randomness in chunk selection."
                )
            self.chunks = self._get_keyframe_chunk_indices()
            # Now, we randomly permute the episodes if shuffle is True
            if shuffle:
                self.current_streaming_chunk_idx = None
                self.current_streaming_frame_idx = None
            else:
                self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
            self.obs_loaders = dict()
            self._should_obs_loaders_reload = True

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        # append metainfo and language annotations
        fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        # TODO: add this back once we have all the language annotations
        # fpaths += [str(self.meta.get_annotation_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """
        Overwrite base method to allow more flexible pattern matching.
        If specific episodes are selected (self.episodes), restrict download to only those
        episode files (data, metainfo, per-episode meta) and, optionally, their videos.
        Otherwise, do coarse filtering based on tasks, cameras, and modalities.
        """
        # Fallback: coarse filtering by tasks/modalities/cameras when episodes are not specified
        allow_patterns = []
        ignore_patterns = []

        # Filter by tasks
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                if self.episodes is not None:
                    for ep_idx in self.episodes:
                        task_id = ep_idx // 10000
                        if task_id == task:
                            allow_patterns.append(f"**/task-{task:04d}/**/episode_{ep_idx:08d}.*")
                            allow_patterns.append(f"**/task-{task:04d}/episode_{ep_idx:08d}.*")
                else:
                    assert False, "Episodes are not specified, so we cannot download the entire task. This is an outstanding TODO"
                    allow_patterns.append(f"**/task-{task:04d}/**")
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(self.task_indices):
                ignore_patterns.append(f"**/task-{task:04d}/**")

        # Filter by modalities/cameras for allow and ignore patterns
        used_modalities = set(self.meta.modalities)
        all_modalities = MODALITY_NAMES
        unused_modalities = all_modalities - used_modalities

        # Ignore unused modalities entirely
        for modality in unused_modalities:
            ignore_patterns.append(f"**/observation.images.{modality}.*/**")

        all_camera_names = set(ROBOT_CAMERA_NAMES["R1Pro"])
        used_camera_names = set(self.meta.camera_names)
        unused_camera_names = all_camera_names - used_camera_names
        for camera in unused_camera_names:
            ignore_patterns.append(f"**/observation.images.*.{camera}/**")

        # Ignore video files when requested
        if not download_videos:
            ignore_patterns.append("videos/")

        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        Overwrite base class to increase max workers to num of CPUs - 2
        """
        logger.info(f"Pulling dataset {self.repo_id} from HuggingFace hub...")
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=os.cpu_count() - 2,
        )

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def __len__(self) -> int:
        """Return the total number of frames available in the chunks."""
        if self._chunk_streaming_using_keyframe:
            # Sum up all chunk lengths
            return sum(chunk_end - chunk_start for chunk_start, chunk_end, _ in self.chunks)
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> dict:
        if not self._chunk_streaming_using_keyframe:
            item = super().__getitem__(idx)
            ep_idx = item["episode_index"].item()
            # Attach per-episode skill prompts (lazy, cached, optional)
            skill_prompts = self._load_episode_skill_prompts(ep_idx)
            # annotations = self._load_episode_annotations(ep_idx)
            if skill_prompts is not None:
                item["skill_prompts"] = skill_prompts
            # if annotations is not None:
            #     item["annotations"] = annotations
            if ep_idx in self.meta.episodes_prompt_variants:
                item["prompt"] = self.meta.episodes_prompt_variants[ep_idx]
            return item
        # Streaming mode: we will load the episode at the current streaming index, and then increment the index for next call
        # Randomize chunk index on first call
        if self.current_streaming_chunk_idx is None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            num_workers = 1 if worker_info is None else worker_info.num_workers
            logger.info(f"INITIALIZING A WORKER WITH WORKER_ID={worker_id} AND RESUME_STEP={self.resume_step}")

            rng = np.random.default_rng(self.seed + worker_id)
            rng.shuffle(self.chunks)
            self.current_streaming_chunk_idx = rng.integers(0, len(self.chunks)).item()
            self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]

            # Fast-forward to resume position if resuming from checkpoint
            if self.resume_step > 0:
                # Calculate how many samples this worker should have processed
                # resume_step is total samples processed globally, distributed among workers in round-robin
                # Each worker gets approximately resume_step / num_workers samples
                samples_to_skip = self.resume_step // num_workers
                if worker_id < (self.resume_step % num_workers):
                    # Some workers may have processed one extra sample due to rounding
                    samples_to_skip += 1

                if samples_to_skip > 0:
                    logger.warning(f"Worker {worker_id}: Resuming from {self.resume_step} global samples, fast-forwarding {samples_to_skip} samples")
                    print(f"Worker {worker_id}: Resuming from {self.resume_step} global samples, fast-forwarding {samples_to_skip} samples")

                    self._fast_forward_frames(samples_to_skip)

                    logger.warning(f"Worker {worker_id}: Resumed at chunk {self.current_streaming_chunk_idx}, frame {self.current_streaming_frame_idx}")
                    print(f"Worker {worker_id}: Resumed at chunk {self.current_streaming_chunk_idx}, frame {self.current_streaming_frame_idx}")

            # if self.checkpoint_dir is not None:
            #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     with open(self.checkpoint_dir / f"chunks_{current_time}_{worker_id}.json", "w") as f:
            #         json.dump({
            #             "current_streaming_chunk_idx": self.current_streaming_chunk_idx,
            #             "current_streaming_frame_idx": self.current_streaming_frame_idx,
            #             "chunks": self.chunks,
            #             "resume_step": self.resume_step,
            #         }, f, indent=4)
        # Current chunk iterated, move to next chunk
        if self.current_streaming_frame_idx >= self.chunks[self.current_streaming_chunk_idx][1]:
            self.current_streaming_chunk_idx += 1
            # All data iterated, restart from beginning
            if self.current_streaming_chunk_idx >= len(self.chunks):
                self.current_streaming_chunk_idx = 0
            self.current_streaming_frame_idx = self.chunks[self.current_streaming_chunk_idx][0]
            self._should_obs_loaders_reload = True
        item = self.hf_dataset[self.current_streaming_frame_idx]
        ep_idx = item["episode_index"].item()

        if self._should_obs_loaders_reload:
            for loader in self.obs_loaders.values():
                loader.close()
            self.obs_loaders = dict()
            # reload video loaders for new episode
            self.current_streaming_episode_idx = ep_idx
            for vid_key in self.meta.video_keys:
                kwargs = {}
                task_id = item["task_index"].item()
                if "seg_instance_id" in vid_key:
                    # load id list
                    with open(
                        self.root / "meta/episodes" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json",
                        "r",
                    ) as f:
                        kwargs["id_list"] = th.tensor(
                            json.load(f)[f"{ROBOT_CAMERA_NAMES['R1Pro'][vid_key.split('.')[-1]]}::unique_ins_ids"]
                        )
                self.obs_loaders[vid_key] = iter(
                    OBS_LOADER_MAP[vid_key.split(".")[2]](
                        data_path=self.root,
                        task_id=task_id,
                        camera_id=vid_key.split(".")[-1],
                        demo_id=f"{ep_idx:08d}",
                        start_idx=self.chunks[self.current_streaming_chunk_idx][2],
                        start_idx_is_keyframe=True,
                        batch_size=1,
                        stride=1,
                        **kwargs,
                    )
                )
            self._should_obs_loaders_reload = False

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(self.current_streaming_frame_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # load visual observations
        for key in self.meta.video_keys:
            item[key] = next(self.obs_loaders[key])[0]

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        # Attach per-episode skill prompts (lazy, cached, optional)
        skill_prompts = self._load_episode_skill_prompts(ep_idx)
        # annotations = self._load_episode_annotations(ep_idx)
        if skill_prompts is not None:
            item["skill_prompts"] = skill_prompts
        # if annotations is not None:
        #     item["annotations"] = annotations
        if ep_idx in self.meta.episodes_prompt_variants:
            item["prompt"] = self.meta.episodes_prompt_variants[ep_idx]

        self.current_streaming_frame_idx += 1

        return item

    def _fast_forward_frames(self, num_samples: int) -> None:
        """
        Fast-forward through the chunk sequence by num_samples frames.
        This is used when resuming from a checkpoint to skip already-processed data.

        Args:
            num_samples: Number of frames to skip forward
        """
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

    def _load_episode_skill_prompts(self, ep_idx: int) -> list[dict] | None:
        """
        Lazy-load and cache per-episode skill prompts based on skill prompts
        Returns skill_prompts_list or None
        """
        if ep_idx not in self._skill_prompts_cache:
            skill_prompts = None
            try:
                skill_prompts_path = self.root / self.meta.get_skill_prompts_path(ep_idx)
                if skill_prompts_path.is_file():
                    with open(skill_prompts_path, "r") as f:
                        skill_prompts = json.load(f)
            except (KeyError, FileNotFoundError, json.JSONDecodeError):
                skill_prompts = None
            self._skill_prompts_cache[ep_idx] = skill_prompts

        return self._skill_prompts_cache[ep_idx]

    # def _load_episode_annotations(self, ep_idx: int) -> dict | None:
    #     """
    #     Lazy-load and cache per-episode annotations based on annotations
    #     Returns annotations or None
    #     """
    #     if ep_idx not in self._annotations_cache:
    #         annotations = None
    #         try:
    #             annotations_path = self.root / self.meta.get_annotation_path(ep_idx)
    #             if annotations_path.is_file():
    #                 with open(annotations_path, "r") as f:
    #                     annotations = json.load(f)
    #         except (KeyError, FileNotFoundError, json.JSONDecodeError):
    #             annotations = None
    #         self._annotations_cache[ep_idx] = annotations

    #     return self._annotations_cache[ep_idx]

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_idx = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": th.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250) -> List[Tuple[int, int, int]]:
        """
        Divide each episode into chunks of data based on GOP of the data (here for B1K, GOP size is 250 frames).
        If frame-level filtering is enabled, only include chunks where more than half the frames are valid.
        If boundary oversampling is enabled, chunks containing boundary frames will be added multiple times.
        If skill oversampling is enabled, chunks containing oversampled skills will be added multiple times.
        Args:
            chunk_size (int): size of each chunk in number of frames. Default is 250 for B1K. Should be the GOP size of the video data.
        Returns:
            List of tuples, where each tuple contains (start_index, end_index, local_start_index) for each chunk.
        """
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()}
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        boundary_chunks_count = 0
        skill_oversampled_chunks_count = 0
        offset = 0
        for L in episode_lengths:  # for each episode
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                # Check if chunk contains boundary frames
                contains_boundary = False
                if self.boundary_frame_indicator is not None:
                    chunk_boundary_indicator = self.boundary_frame_indicator[offset + ls:offset + le]
                    # At least 10% of the chunk must contain boundary points
                    contains_boundary = sum(chunk_boundary_indicator) > chunk_size * 0.2

                # Get the maximum oversampling factor from skills in this chunk
                max_skill_oversample_factor = 1  # default: normal inclusion
                if self.oversampled_skill_indicator is not None:
                    chunk_skill_indicator = self.oversampled_skill_indicator[offset + ls:offset + le]
                    # Get the maximum oversampling factor in this chunk
                    max_skill_oversample_factor = max(chunk_skill_indicator, default=1)

                # Calculate base duplication factor from boundaries and skills
                duplication_factor = max(
                    self.boundary_oversampling_factor if contains_boundary else 1,
                    max_skill_oversample_factor
                )

                # Track chunks with oversampling for logging
                if contains_boundary:
                    boundary_chunks_count += 1
                if max_skill_oversample_factor > 1:
                    skill_oversampled_chunks_count += 1

                # Apply filtering only if no oversampling is active
                if duplication_factor == 1 and self.valid_frame_mask is not None:
                    chunk_mask = self.valid_frame_mask[offset + ls:offset + le]
                    # Skip chunk if less than 50% valid frames
                    if sum(chunk_mask) <= (chunk_size / 2):
                        duplication_factor = 0

                # Add chunk the appropriate number of times
                for _ in range(duplication_factor):
                    chunks.append((offset + ls, offset + le, ls))

            offset += L

        if self.valid_frame_mask is not None:
            logger.info(f"Chunk-level filtering: kept {len(chunks)} chunk instances (some chunks may be duplicated)")

        if self.boundary_frame_indicator is not None:
            logger.info(f"Boundary oversampling: {boundary_chunks_count} unique chunks contain boundaries and are repeated up to {self.boundary_oversampling_factor}x")

        if self.oversampled_skill_indicator is not None:
            logger.info(f"Skill oversampling: {skill_oversampled_chunks_count} unique chunks contain oversampled skills")

        logger.info(f"Total chunks: {len(chunks)}")

        return chunks

    def _build_undersampled_frame_mask(self) -> None:
        """
        Build a frame-level validity mask based on skill annotations and undersampled skill descriptions.
        For each episode, loads the annotations and marks frames as valid/invalid based on 
        probabilistic undersampling of skills according to undersampled_skill_descriptions dict.
        Skills not in the dict are assumed to have probability 1.0 (always included).
        Episodes where annotations cannot be loaded will have all their frames marked as invalid.

        This method is called BEFORE chunks are created, allowing _get_keyframe_chunk_indices
        to filter out entire chunks that contain undersampled frames.

        Uses self.seed for RNG reproducibility.
        """
        logger.info(f"Building undersampled frame mask with {len(self.undersampled_skill_descriptions)} undersampled skill descriptions: {self.undersampled_skill_descriptions}")

        # Create RNG for reproducible probabilistic sampling
        rng = np.random.default_rng(self.seed)

        # Build a mapping of global frame index to whether it should be included
        valid_frame_mask = []

        for ep_idx in self.episodes:
            task_idx = ep_idx // 10000
            ep_length = self.meta.episodes[ep_idx]["length"]

            # Try to load annotations for this episode
            annotation_path = self.root / "annotations" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.json"
            try:
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)
                skill_annotations = annotations.get("skill_annotation", [])

                # Create a boolean mask for this episode, default to True for all frames
                episode_mask = [True] * ep_length

                # Mark frames as valid based on probabilistic undersampling
                for skill in skill_annotations:
                    # Get skill description
                    skill_desc = skill["skill_description"][0]

                    # Decide whether to include this skill based on probability
                    if skill_desc not in self.undersampled_skill_descriptions:
                        continue

                    # Get inclusion probability
                    inclusion_prob = self.undersampled_skill_descriptions[skill_desc]

                    if rng.random() < inclusion_prob:
                        # Skip removing this skill i.e. keep it
                        continue

                    # this can be either a list of 2 integers (start_frame, end_frame) or a list of lists of 2 integers
                    frame_duration_lists = skill["frame_duration"]
                    if isinstance(frame_duration_lists, list) and len(frame_duration_lists) == 2 and all(isinstance(x, int) for x in frame_duration_lists):
                        frame_duration_lists = [frame_duration_lists]

                    # each element here is guaranteed to be a list of 2 integers
                    for frame_duration_list in frame_duration_lists:
                        start_frame, end_frame = frame_duration_list
                        for frame_idx in range(start_frame, end_frame):
                            if frame_idx < ep_length:
                                # remove this series of frames from the training data
                                episode_mask[frame_idx] = False

                valid_frame_mask.extend(episode_mask)

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load annotations for episode {ep_idx}: {e}. Marking all frames as invalid.")
                # Mark all frames in this episode as invalid
                valid_frame_mask.extend([False] * ep_length)

        # Store the valid frame mask for use in _get_keyframe_chunk_indices
        self.valid_frame_mask = valid_frame_mask

        # Log overall statistics
        total_frames = len(valid_frame_mask)
        valid_frames = sum(valid_frame_mask)
        logger.info(f"Frame-level mask built: {valid_frames}/{total_frames} frames marked as valid ({100*valid_frames/total_frames:.1f}%). NOTE: this does not account for boundary frames!")

    def _build_oversampled_skill_indicator(self) -> None:
        """
        Build a frame-level indicator marking frames belonging to oversampled skills.
        For each episode, loads the annotations and marks frames that belong to skills in
        oversampled_skill_descriptions dict. These frames will be used by _get_keyframe_chunk_indices
        to duplicate chunks containing oversampled skills.

        Each frame is mapped to the maximum oversampling factor of any skill it belongs to.
        If a frame belongs to multiple oversampled skills, the highest factor is used.
        1 means normal inclusion (not oversampled), >1 means oversample by that factor.
        """
        logger.info(f"Building oversampled skill indicator with {len(self.oversampled_skill_descriptions)} oversampled skill descriptions: {self.oversampled_skill_descriptions}")

        # Build a mapping of global frame index to oversampling factor
        # 1 means normal inclusion, >1 means oversample by that factor
        oversampled_skill_indicator = []

        for ep_idx in self.episodes:
            task_idx = ep_idx // 10000
            ep_length = self.meta.episodes[ep_idx]["length"]

            # Try to load annotations for this episode
            annotation_path = self.root / "annotations" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.json"
            try:
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)
                skill_annotations = annotations.get("skill_annotation", [])

                # Create an indicator for this episode, default to 1 (normal inclusion)
                episode_indicator = [1] * ep_length

                # Mark frames belonging to oversampled skills
                for skill in skill_annotations:
                    skill_desc = skill["skill_description"][0]

                    # Check if this skill should be oversampled
                    if skill_desc not in self.oversampled_skill_descriptions:
                        continue

                    # Get oversampling factor
                    oversample_factor = self.oversampled_skill_descriptions[skill_desc]

                    # Handle both simple and complex frame_duration formats
                    frame_duration_lists = skill["frame_duration"]
                    if isinstance(frame_duration_lists, list) and len(frame_duration_lists) == 2 and all(isinstance(x, int) for x in frame_duration_lists):
                        frame_duration_lists = [frame_duration_lists]

                    # Mark all frames in this skill with the oversampling factor
                    # If a frame already has a factor, keep the maximum
                    for frame_duration_list in frame_duration_lists:
                        start_frame, end_frame = frame_duration_list
                        for frame_idx in range(start_frame, end_frame):
                            if frame_idx < ep_length:
                                episode_indicator[frame_idx] = max(episode_indicator[frame_idx], oversample_factor)

                oversampled_skill_indicator.extend(episode_indicator)

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load annotations for episode {ep_idx}: {e}. No oversampling for this episode.")
                # Mark all frames as normal (not oversampled)
                oversampled_skill_indicator.extend([1] * ep_length)

        # Store the oversampled skill indicator
        self.oversampled_skill_indicator = oversampled_skill_indicator

        # Log statistics
        total_frames = len(oversampled_skill_indicator)
        oversampled_frames = sum(1 for x in oversampled_skill_indicator if x > 1)
        if total_frames > 0:
            logger.info(f"Oversampled skill indicator built: {oversampled_frames}/{total_frames} frames marked for oversampling ({100*oversampled_frames/total_frames:.1f}%)")

    def _build_boundary_frame_indicator(self) -> None:
        """
        Build a frame-level indicator marking frames near skill boundaries.
        Frames within boundary_window_frames of any skill transition are marked as True.
        This allows _get_keyframe_chunk_indices to oversample chunks containing these critical transitions.

        A skill boundary is defined as the frame where one skill ends and another begins.
        For each boundary, we mark frames in the window [boundary - window : boundary + window].
        """
        logger.info(f"Building boundary frame indicator with window={self.boundary_window_frames} frames, oversampling factor={self.boundary_oversampling_factor}")

        # Build a boolean indicator for boundary frames
        boundary_frame_indicator = []
        boundary_count = 0

        for ep_idx in self.episodes:
            task_idx = ep_idx // 10000
            ep_length = self.meta.episodes[ep_idx]["length"]

            # Try to load annotations for this episode
            annotation_path = self.root / "annotations" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.json"
            try:
                with open(annotation_path, "r") as f:
                    annotations = json.load(f)
                skill_annotations = annotations.get("skill_annotation", [])

                # Create a boolean indicator for this episode, default to False
                episode_indicator = [False] * ep_length

                # Mark frames near skill boundaries
                # Sort skills by their start frame to ensure correct ordering
                sorted_skills = sorted(skill_annotations, key=lambda s: s["frame_duration"][0] if isinstance(s["frame_duration"][0], int) else s["frame_duration"][0][0])

                for i in range(len(sorted_skills) - 1):
                    current_skill = sorted_skills[i]
                    next_skill = sorted_skills[i + 1]

                    # Get end frame(s) of current skill
                    current_duration = current_skill["frame_duration"]
                    boundary_frames = []

                    if isinstance(current_duration, list) and len(current_duration) == 2 and all(isinstance(x, int) for x in current_duration):
                        # Simple case: [start, end]
                        boundary_frames = [current_duration[1]]
                    elif isinstance(current_duration, list) and isinstance(current_duration[0], (list, tuple)):
                        # Complex case: [[start1, end1], [start2, end2], ...] or [(start1, end1), ...]
                        # Each segment's end is a boundary point
                        boundary_frames = [segment[1] for segment in current_duration]
                    else:
                        continue

                    # Mark frames in window around each boundary
                    for boundary_frame in boundary_frames:
                        window_start = max(0, boundary_frame - self.boundary_window_frames)
                        window_end = min(ep_length, boundary_frame + self.boundary_window_frames + 1)

                        for frame_idx in range(window_start, window_end):
                            episode_indicator[frame_idx] = True

                        boundary_count += 1

                boundary_frame_indicator.extend(episode_indicator)

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load annotations for episode {ep_idx}: {e}. No boundaries marked for this episode.")
                # Mark all frames as non-boundary
                boundary_frame_indicator.extend([False] * ep_length)

        # Store the boundary frame indicator
        self.boundary_frame_indicator = boundary_frame_indicator

        # Log statistics
        total_frames = len(boundary_frame_indicator)
        boundary_frames = sum(boundary_frame_indicator)
        logger.info(f"Boundary frame indicator built: {boundary_frames}/{total_frames} frames marked as boundary ({100*boundary_frames/total_frames:.1f}%), {boundary_count} boundaries detected")

    def close(self) -> None:
        """
        Release any underlying resources (e.g., video loaders) held by this dataset.
        Safe to call multiple times.
        """
        try:
            if hasattr(self, "obs_loaders") and isinstance(self.obs_loaders, dict):
                for loader in list(self.obs_loaders.values()):
                    try:
                        if hasattr(loader, "close"):
                            loader.close()
                    except Exception:
                        pass
                self.obs_loaders = dict()
        except Exception:
            # Best-effort cleanup
            pass

    def __del__(self):
        # Ensure resources are cleaned up if the object is garbage-collected
        self.close()

class BehaviorLerobotDatasetMetadata(LeRobotDatasetMetadata):
    """
    BehaviorLerobotDatasetMetadata extends LeRobotDatasetMetadata with the following customizations:
        1. Restricts the set of allowed modalities to {"rgb", "depth", "seg_instance_id"}.
        2. Restricts the set of allowed camera names to those defined in ROBOT_CAMERA_NAMES["R1Pro"].
        3. Provides a filtered view of dataset features, including only those corresponding to the selected modalities and camera names.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        # === Customized arguments for BehaviorLeRobotDataset ===
        tasks: Iterable[str] = None,
        modalities: Iterable[str] = None,
        cameras: Iterable[str] = None,
    ):
        # ========== Customizations ==========
        self.task_name_candidates = set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        self.modalities = set(modalities)
        self.camera_names = set(cameras)
        assert self.modalities.issubset(
            MODALITY_NAMES
        ), f"Modalities must be a subset of {MODALITY_NAMES}, but got {self.modalities}"
        assert self.camera_names.issubset(
            ROBOT_CAMERA_NAMES["R1Pro"]
        ), f"Camera names must be a subset of {ROBOT_CAMERA_NAMES['R1Pro']}, but got {self.camera_names}"
        # ===================================

        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/**", ignore_patterns="meta/episodes/**")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(self.root)
        # filter based on self.task_name_candidates
        valid_task_indices = [idx for idx, name in self.task_names.items() if name in self.task_name_candidates]
        self.task_names = set([self.task_names[idx] for idx in valid_task_indices])
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        logger.info(f"Loaded metadata for {len(self.episodes)} episodes.")

        self.episodes_prompt_variants = self.load_episodes_prompt_variants(self.root)
        logger.info(f"Loaded {len(self.episodes_prompt_variants)} episodes prompt variants.")

    def load_tasks(self, local_dir: Path) -> tuple[dict, dict]:
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {item["task_index"]: item["task_name"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
        task_to_task_index = {task: task_index for task_index, task in tasks.items()}
        return tasks, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path) -> dict:
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_episodes_prompt_variants(self, local_dir: Path) -> dict:
        episodes_prompt_variants = load_jsonlines(local_dir / EPISODES_PROMPT_VARIANTS_PATH)
        logger.info(f"Loaded {len(episodes_prompt_variants)} episodes prompt variants.")
        return {
            item["episode_index"]: item["tasks"][0]
            for item in sorted(episodes_prompt_variants, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks and "tasks" in item and len(item["tasks"]) > 0
        }

    def load_stats(self, local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path) -> dict:
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def get_annotation_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_skill_prompts_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.skill_prompts_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Formattable string for the annotation files."""
        return self.info["annotation_path"]

    @property
    def metainfo_path(self) -> str | None:
        """Formattable string for the metainfo files."""
        return self.info["metainfo_path"]

    @property
    def skill_prompts_path(self) -> str | None:
        """Formattable string for the skill prompts files."""
        if self.info["annotation_path"] is not None:
            return self.info["annotation_path"].replace("annotations/", "skill_prompts/")
        return None

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        features = dict()
        # pop not required features
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features
