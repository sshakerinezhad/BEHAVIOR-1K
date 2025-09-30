import logging
import torch as th
from typing import Optional

from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


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

    policy =_policy_config.create_trained_policy(
        _config.get_config(policy_config), policy_dir
    )
    policy_metadata = policy.metadata
    policy = _policy.PolicyRecorder(policy, "policy_records")
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
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        if policy_config is not None and policy_dir is not None and task_name is not None:
            self.policy = load_policy(policy_config, policy_dir, task_name)
        else:
            self.policy = None  # To be set later

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            return th.zeros(self.action_dim, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()


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
        return self.policy.act(obs).detach().cpu()

    def reset(self) -> None:
        self.policy.reset()
