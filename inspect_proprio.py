"""
The purpose of this script is to inspect the proprioception observations from the dataset vs from
the simulator environment and see if there is some minor, easily correctable discrepancy between them.
"""

import torch.nn.functional as F

from test_inference_equality import load_pickle

def main():
    datapoint_0 = load_pickle("curr_datapoint.pkl")
    datapoint_49 = load_pickle("curr_datapoint_49.pkl")
    datapoint_240 = load_pickle("curr_datapoint_240.pkl")

    obs_0 = load_pickle("obs_0.pkl")
    obs_49 = load_pickle("obs_49.pkl")
    obs_240 = load_pickle("obs_240.pkl")

    proprio_gt_0 = datapoint_0["observation.state"]
    proprio_gt_49 = datapoint_49["observation.state"]
    proprio_gt_240 = datapoint_240["observation.state"]

    proprio_obs_0 = obs_0["robot_r1::proprio"]
    proprio_obs_49 = obs_49["robot_r1::proprio"]
    proprio_obs_240 = obs_240["robot_r1::proprio"]

    print("F.cosine_similarity(proprio_gt_0, proprio_obs_0)", F.cosine_similarity(proprio_gt_0, proprio_obs_0, dim=0))
    print("F.cosine_similarity(proprio_gt_49, proprio_obs_49)", F.cosine_similarity(proprio_gt_49, proprio_obs_49, dim=0))
    print("F.cosine_similarity(proprio_gt_240, proprio_obs_240)", F.cosine_similarity(proprio_gt_240, proprio_obs_240, dim=0))

    print("F.mse_loss(proprio_gt_0, proprio_obs_0)", F.mse_loss(proprio_gt_0, proprio_obs_0))
    print("F.mse_loss(proprio_gt_49, proprio_obs_49)", F.mse_loss(proprio_gt_49, proprio_obs_49))
    print("F.mse_loss(proprio_gt_240, proprio_obs_240)", F.mse_loss(proprio_gt_240, proprio_obs_240))

    breakpoint()


if __name__ == "__main__":
    main()
