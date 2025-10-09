"""
The goal of this script is to analyze the difference between input observation `obs` dicts from the
simulation environment vs from the training dataset `curr_datapoint`. This will tell us if the simulation environment
has any bugs or major differences from the training dataset.

We will compare the inputs themselves as well as the different outputs of the finetuned policy
model on these different inputs.

We suspect that the proprioceptive state is the main source of difference between the two, but
this remains to be seen.
"""

from collections import defaultdict
import os

import torch.nn.functional as F
import torch

from omnigibson.learning.policies import load_policy, LookupPolicy
from openpi_client.image_tools import resize_with_pad

from test_inference_equality import load_pickle, IMAGE_KEYS
from check_action_similarity import get_obs_from_datapoint


OBS_DIR = "obs_from_eval/"
STEP_START = 1000
STEP_COUNT = 2000
STEP_INTERVAL = 100
steps = list(range(STEP_START, STEP_COUNT, STEP_INTERVAL))
print(f"Steps: {steps}")

def main():
    task_instance = "1"
    policy = load_policy(
        policy_config="pi05_b1k",
        policy_dir="/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/",
        task_name="turning_on_radio",
    )
    dataset_policy = LookupPolicy(
        policy_config="pi05_b1k",
        task_name="turning_on_radio",
    )
    dataset_policy.set_task_instance(task_instance)
    curr_obs_dir = os.path.join(OBS_DIR, task_instance)

    diff_left = defaultdict(list)
    diff_right = defaultdict(list)
    diff_head = defaultdict(list)
    diff_proprio = defaultdict(list)
    diff_cam_rel_poses = defaultdict(list)
    diff_action_gt_vs_dp = defaultdict(list)
    diff_action_gt_vs_sim = defaultdict(list)
    diff_action_gt_vs_sim_with_proprio_from_dp = defaultdict(list)

    for step in steps:
        obs = load_pickle(os.path.join(curr_obs_dir, f"{step}.pkl"))
        obs_from_datapoint = get_obs_from_datapoint(dataset_policy.current_datapoint)

        for img_key in IMAGE_KEYS:
            obs[img_key] = obs[img_key][:, :, :3]
            obs_from_datapoint[img_key] = resize_with_pad(obs_from_datapoint[img_key], 224, 224)

        # Left realsense
        diff_left["cosine_similarity"].append(
            F.cosine_similarity(
                obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]).type(torch.float64)
            ).mean()
        )
        diff_left["mae_loss"].append(
            F.l1_loss(
                obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]).type(torch.float64)
            ).item()
        )

        # Right realsense
        diff_right["cosine_similarity"].append(
            F.cosine_similarity(
                obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]).type(torch.float64)
            ).mean()
        )
        diff_right["mae_loss"].append(
            F.l1_loss(
                obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]).type(torch.float64)
            ).item()
        )

        # Head zed
        diff_head["cosine_similarity"].append(
            F.cosine_similarity(
                obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:zed_link:Camera:0::rgb"]).type(torch.float64)
            ).mean()
        )
        diff_head["mae_loss"].append(
            F.l1_loss(
                obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::robot_r1:zed_link:Camera:0::rgb"]).type(torch.float64)
            ).item()
        )

        # Proprio
        diff_proprio["cosine_similarity"].append(
            F.cosine_similarity(
                obs["robot_r1::proprio"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::proprio"]).type(torch.float64),
                dim=0
            ).item()
        )
        diff_proprio["mse_loss"].append(
            F.mse_loss(
                obs["robot_r1::proprio"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::proprio"]).type(torch.float64),
            ).item()
        )

        # Cam rel poses
        diff_cam_rel_poses["cosine_similarity"].append(
            F.cosine_similarity(
                obs["robot_r1::cam_rel_poses"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::cam_rel_poses"]).type(torch.float64),
                dim=0,
            ).item()
        )
        diff_cam_rel_poses["mse_loss"].append(
            F.mse_loss(
                obs["robot_r1::cam_rel_poses"].type(torch.float64),
                torch.from_numpy(obs_from_datapoint["robot_r1::cam_rel_poses"]).type(torch.float64)
            ).item()
        )

        action_gt = dataset_policy.current_datapoint["action"][0]

        action_dp = policy.act(obs_from_datapoint).detach().cpu()[0]
        diff_action_gt_vs_dp["cosine_similarity"].append(
            F.cosine_similarity(action_gt, action_dp, dim=0).item()
        )
        diff_action_gt_vs_dp["mse_loss"].append(
            F.mse_loss(action_gt, action_dp).item()
        )

        action_sim = policy.act(obs).detach().cpu()[0]
        diff_action_gt_vs_sim["cosine_similarity"].append(
            F.cosine_similarity(action_gt, action_sim, dim=0).item()
        )
        diff_action_gt_vs_sim["mse_loss"].append(
            F.mse_loss(action_gt, action_sim).item()
        )

        obs_from_datapoint_with_proprio_from_dp = {
            **obs_from_datapoint,
            "robot_r1::proprio": obs_from_datapoint["robot_r1::proprio"]
        }
        action_sim_with_proprio_from_dp = policy.act(obs_from_datapoint_with_proprio_from_dp).detach().cpu()[0]
        diff_action_gt_vs_sim_with_proprio_from_dp["cosine_similarity"].append(
            F.cosine_similarity(action_gt, action_sim_with_proprio_from_dp, dim=0).item()
        )
        diff_action_gt_vs_sim_with_proprio_from_dp["mse_loss"].append(
            F.mse_loss(action_gt, action_sim_with_proprio_from_dp).item()
        )

    print("Left realsense cosine similarity", torch.tensor(diff_left["cosine_similarity"]).mean())
    print("Left realsense mae loss", torch.tensor(diff_left["mae_loss"]).mean())
    print("")
    print("Right realsense cosine similarity", torch.tensor(diff_right["cosine_similarity"]).mean())
    print("Right realsense mae loss", torch.tensor(diff_right["mae_loss"]).mean())
    print("")
    print("Head zed cosine similarity", torch.tensor(diff_head["cosine_similarity"]).mean())
    print("Head zed mae loss", torch.tensor(diff_head["mae_loss"]).mean())
    print("")
    print("Proprio cosine similarity", torch.tensor(diff_proprio["cosine_similarity"]).mean())
    print("Proprio mse loss", torch.tensor(diff_proprio["mse_loss"]).mean())
    print("")
    print("Cam rel poses cosine similarity", torch.tensor(diff_cam_rel_poses["cosine_similarity"]).mean())
    print("Cam rel poses mse loss", torch.tensor(diff_cam_rel_poses["mse_loss"]).mean())
    print("")
    print("Action gt vs datapoint cosine similarity", torch.tensor(diff_action_gt_vs_dp["cosine_similarity"]).mean())
    print("Action gt vs datapoint mse loss", torch.tensor(diff_action_gt_vs_dp["mse_loss"]).mean())
    print("")
    print("Action gt vs simulation cosine similarity", torch.tensor(diff_action_gt_vs_sim["cosine_similarity"]).mean())
    print("Action gt vs simulation mse loss", torch.tensor(diff_action_gt_vs_sim["mse_loss"]).mean())
    print("")
    print("Action gt vs simulation but with proprio from datapoint cosine similarity", torch.tensor(diff_action_gt_vs_sim_with_proprio_from_dp["cosine_similarity"]).mean())
    print("Action gt vs simulation but with proprio from datapoint mse loss", torch.tensor(diff_action_gt_vs_sim_with_proprio_from_dp["mse_loss"]).mean())

    breakpoint()


if __name__ == "__main__":
    main()
