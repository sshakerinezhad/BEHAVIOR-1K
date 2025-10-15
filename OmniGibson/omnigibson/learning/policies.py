import json
import logging
import os
import pickle
import torch as th
from typing import Optional

from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata, BehaviorLeRobotDataset

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

OBS_TO_DP_MAPPING = {
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": "observation.images.rgb.left_wrist",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": "observation.images.rgb.right_wrist",
    "robot_r1::robot_r1:zed_link:Camera:0::rgb": "observation.images.rgb.head",
    "robot_r1::proprio": "observation.state",
    "robot_r1::cam_rel_poses": "observation.cam_rel_poses",
}

IMAGE_KEYS = [
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
]

__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def convert_obs_to_numpy(obs):
    return {k: v.numpy() for k, v in obs.items()}


def get_obs_from_datapoint(datapoint):
    obs = {
        "robot_r1::proprio": datapoint["observation.state"],
        "robot_r1::cam_rel_poses": datapoint["observation.cam_rel_poses"],
    }
    for img_key in IMAGE_KEYS:
        obs[img_key] = datapoint[OBS_TO_DP_MAPPING[img_key]].permute(1, 2, 0)
    return convert_obs_to_numpy(obs)


def load_policy(policy_config: str, policy_dir: str, task_name: str):
    dataset_root = "/scr/behavior/2025-challenge-demos"
    metadata = BehaviorLerobotDatasetMetadata(
        repo_id="behavior-1k/2025-challenge-demos",
        root=dataset_root,
        tasks=[task_name] if task_name else "turning_on_radio",
        modalities=[],
        cameras=[],
    )
    prompt = list(metadata.tasks.values())[0]
    # log the prompt used
    logging.info(f"Using prompt: {prompt}")
    logging.info(f"Using policy config: {policy_config}")
    logging.info(f"Using policy dir: {policy_dir}")
    policy =_policy_config.create_trained_policy(
        _config.get_config(policy_config), policy_dir
    )
    policy_metadata = policy.metadata
    # policy = _policy.PolicyRecorder(policy, "policy_records")
    policy = B1KPolicyWrapper(policy, text_prompt=prompt)
    return policy


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(
        self,
        *args,
        action_dim: Optional[int] = None,
        policy_config: Optional[str] = None,
        policy_dir: Optional[str] = None,
        task_name: Optional[str] = None,
        use_dataset_inputs: Optional[bool] = False,
        use_dataset_inputs_proprio_only: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        self.dataset_policy = None
        self.use_dataset_inputs = use_dataset_inputs
        self.use_dataset_inputs_proprio_only = use_dataset_inputs_proprio_only
        if policy_config is not None and policy_dir is not None and task_name is not None:
            if self.use_dataset_inputs or self.use_dataset_inputs_proprio_only:
                self.dataset_policy = LookupPolicy(policy_config=policy_config, task_name=task_name)
            self.policy = load_policy(policy_config, policy_dir, task_name)
        else:
            self.policy = None  # To be set later

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        out_path = f"./obs_from_eval_v2/{self.task_instance}/{self.step_count}.pkl"
        save_pickle(obs, out_path)
        if self.policy is not None:
            if self.use_dataset_inputs_proprio_only:
                obs_with_proprio_from_dp = {
                    **obs,
                    "robot_r1::proprio": self.dataset_policy.current_datapoint[OBS_TO_DP_MAPPING["robot_r1::proprio"]]
                }
                out = self.policy.act(obs_with_proprio_from_dp).detach().cpu()[0]
                self.dataset_policy.run_iterator_step()
            elif self.use_dataset_inputs:
                obs_from_datapoint = get_obs_from_datapoint(self.dataset_policy.current_datapoint)
                out = self.policy.act(obs_from_datapoint).detach().cpu()[0]
                self.dataset_policy.run_iterator_step()
            else:
                obs = convert_obs_to_numpy(obs)
                out = self.policy.act(obs).detach().cpu()[0]
            self.step_count += 1
            return out
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.task_instance = None
        self.step_count = 0
        if self.dataset_policy is not None:
            self.dataset_policy.reset()

    def set_task_instance(self, idx: str) -> None:
        self.task_instance = idx
        if self.dataset_policy is not None:
            self.dataset_policy.set_task_instance(idx)
        os.makedirs(f"./obs_from_eval_v2/{self.task_instance}", exist_ok=True)


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.policy = WebsocketClientPolicy(host=host, port=port)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        out = self.policy.act(obs).detach().cpu()
        print(out.shape)
        return out

    def reset(self) -> None:
        self.policy.reset()

    def set_task_instance(self, idx: str) -> None:
        pass


class LookupPolicy:
    """
    Lookup policy that looks up the correct action from a lookup table.

    Only for training set episodes.
    """

    def __init__(
        self,
        *args,
        policy_config: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs
    ) -> None:
        config = _config.get_config(policy_config)
        data_config = config.data.create(config.assets_dirs, config.model)
        self.dataset = iter(
            BehaviorLeRobotDataset(
                repo_id=data_config.repo_id,
                root=data_config.behavior_dataset_root,
                tasks=[task_name],
                modalities=["rgb"],
                local_only=False,
                delta_timestamps={
                    key: [t / 30.0 for t in range(config.model.action_horizon)] for key in data_config.action_sequence_keys
                },
                episodes=list(range(190)),
                chunk_streaming_using_keyframe=True,
                shuffle=False,
            )
        )
        self.task_instance = None
        self.current_datapoint = None

    def run_iterator_step(self):
        self.current_datapoint = next(self.dataset)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # We use 0 as the index because the index is irrelevant when chunk_streaming_using_keyframe=True
        curr_datapoint = self.current_datapoint

        # Update the current datapoint for the next call
        self.run_iterator_step()

        return curr_datapoint["action"][0]

    def reset(self) -> None:
        self.task_instance = None
        self.current_datapoint = None

    def set_task_instance(self, idx: str) -> None:
        self.task_instance = idx
        self.run_iterator_step()
        while str(int((self.current_datapoint["episode_index"] // 10) % 1e3)) != idx:
            print(f"[Seeking task instance {idx}], skipping episode_index={self.current_datapoint['episode_index']}, index={self.current_datapoint['index']}")
            self.run_iterator_step()
