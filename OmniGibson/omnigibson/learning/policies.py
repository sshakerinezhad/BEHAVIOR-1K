import json
import dataclasses
import logging
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
    return {k: v.numpy() if hasattr(v, "numpy") else v for k, v in obs.items()}


def get_obs_from_datapoint(datapoint):
    obs = {
        "robot_r1::proprio": datapoint["observation.state"],
        "robot_r1::cam_rel_poses": datapoint["observation.cam_rel_poses"],
    }
    for img_key in IMAGE_KEYS:
        obs[img_key] = datapoint[OBS_TO_DP_MAPPING[img_key]].permute(1, 2, 0)
    return convert_obs_to_numpy(obs)


def load_policy(policy_config: str, policy_dir: str, inf_time_proprio_dropout: float):
    logging.info(f"Using policy config: {policy_config}")
    logging.info(f"Using policy dir: {policy_dir}")
    basic_config = _config.get_config(policy_config)
    updated_config_model = dataclasses.replace(
        basic_config.model,
        proprio_dropout_dropout_whole_proprio_pct=inf_time_proprio_dropout
    )
    updated_config = dataclasses.replace(
        basic_config,
        model=updated_config_model
    )
    policy = _policy_config.create_trained_policy(updated_config, policy_dir)
    # policy = _policy.PolicyRecorder(policy, "policy_records")
    policy = B1KPolicyWrapper(policy, config=updated_config)
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
        prompt: Optional[str] = None,
        inf_time_proprio_dropout: Optional[float] = 0.0,
        n_ds_steps: Optional[int] = 0,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        self.dataset_policy = None
        self.use_dataset_inputs = use_dataset_inputs
        self.use_dataset_inputs_proprio_only = use_dataset_inputs_proprio_only
        self.n_ds_steps = n_ds_steps
        if policy_config is not None and policy_dir is not None and task_name is not None:
            if self.use_dataset_inputs or self.use_dataset_inputs_proprio_only or self.n_ds_steps > 0:
                self.dataset_policy = LookupPolicy(policy_config=policy_config, task_name=task_name)
            self.policy = load_policy(policy_config, policy_dir, inf_time_proprio_dropout)
        else:
            self.policy = None  # To be set later
        self.prompt = prompt

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.prompt is not None:
            obs["prompt"] = self.prompt
        if self.policy is not None:
            if self.use_dataset_inputs_proprio_only:
                obs_with_proprio_from_dp = {
                    **obs,
                    "robot_r1::proprio": self.dataset_policy.get_current_datapoint()[OBS_TO_DP_MAPPING["robot_r1::proprio"]]
                }
                out = self.policy.act(obs_with_proprio_from_dp).detach().cpu()
            elif self.use_dataset_inputs:
                obs_from_datapoint = get_obs_from_datapoint(self.dataset_policy.get_current_datapoint())
                out = self.policy.act(obs_from_datapoint).detach().cpu()
            else:
                if self.n_ds_steps > 0 and self.step_count < self.n_ds_steps:
                    self.step_count += 1
                    return self.dataset_policy.forward({})
                obs = convert_obs_to_numpy(obs)
                out = self.policy.act(obs).detach().cpu()
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

    def set_task_instance(self, idx: int) -> None:
        self.task_instance = idx
        if self.dataset_policy is not None:
            self.dataset_policy.set_task_instance(idx)


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.prompt = prompt

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        # convert observation to numpy
        obs = torch_to_numpy(obs)
        if self.prompt is not None:
            obs["prompt"] = self.prompt
        return self.policy.act(obs).detach().cpu()

    def reset(self) -> None:
        self.policy.reset()

    def set_task_instance(self, idx: int) -> None:
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
        self.config = _config.get_config(policy_config)
        self.data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        self.dataset = None
        self.task_instance = None
        self.task_name = task_name

    def get_current_datapoint(self):
        return next(self.dataset)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        curr_datapoint = self.get_current_datapoint()
        return curr_datapoint["action"][0]

    def reset(self) -> None:
        self.task_instance = None

    def set_task_instance(self, idx: int) -> None:
        self.task_instance = idx
        logging.info(f"Now loading episode {idx} for task {self.task_name}")
        self.dataset = iter(
            BehaviorLeRobotDataset(
                repo_id=self.data_config.repo_id,
                root=self.data_config.behavior_dataset_root,
                tasks=[self.task_name],
                modalities=["rgb"],
                local_only=False,
                delta_timestamps={
                    key: [t / 30.0 for t in range(self.config.model.action_horizon)] for key in self.data_config.action_sequence_keys
                },
                episodes=[idx],
                chunk_streaming_using_keyframe=True,
                shuffle=False,
            )
        )
