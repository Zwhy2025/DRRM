import os
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel

from drrm.common.pytorch_util import dict_apply


def load_policy(ckp_path: str, use_ckp_code: bool = True):
    if use_ckp_code:
        policy_model = AutoModel.from_pretrained(ckp_path, trust_remote_code=True)
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    else:
        config = AutoConfig.from_pretrained(ckp_path, trust_remote_code=True)
        ConfigClass = hydra.utils.get_class(config.pkg_map["AutoConfig"])
        PolicyClass = hydra.utils.get_class(config.pkg_map["AutoModel"])
        config = ConfigClass.from_pretrained(ckp_path)
        policy_model = PolicyClass(config)
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    return policy_model


class VODPRunner:
    def __init__(self, n_obs_steps: int, n_action_steps: int = 8):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs = deque(maxlen=n_obs_steps + 1)

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs: Dict[str, Any]):
        self.obs.append(current_obs)

    def _stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def _get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"
        result = dict()
        for key in self.obs[0].keys():
            result[key] = self._stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps,
            )
        return result

    def get_action(self, policy, observation=None):
        device = policy.device
        if observation is not None:
            self.obs.append(observation)
        obs = self._get_n_steps_obs()
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            obs_dict_input = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
            action_dict = policy.predict_action(obs_dict_input)
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").float().numpy())
        action = np_action_dict["action"].squeeze(0)
        return action


class VODPInference:
    def __init__(self, checkpoint_dir: str, mixed_precision: str = "bf16", device: str = "cuda", use_ckp_code: bool = False):
        if mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.policy = load_policy(checkpoint_dir, use_ckp_code=use_ckp_code)
        self.policy.eval()
        self.policy.to(device)

        self.runner = VODPRunner(n_obs_steps=self.policy.n_obs_steps)

    def update_obs(self, observation: Dict[str, Any]):
        self.runner.update_obs(observation)

    def predict_action(self, observation=None):
        device = str(self.policy.device)
        if self.dtype == torch.float32:
            return self.runner.get_action(self.policy, observation)
        with torch.autocast(device_type=device, dtype=self.dtype):
            return self.runner.get_action(self.policy, observation)

    def get_action(self, observation=None):
        return self.predict_action(observation)

    def reset(self):
        self.runner.reset_obs()

    def get_last_obs(self):
        return self.runner.obs[-1]

    @classmethod
    def from_config(cls, cfg):
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        return cls(
            checkpoint_dir=cfg.checkpoint_dir,
            mixed_precision=cfg.mixed_precision,
            device="cuda",
            use_ckp_code=False,
        )


__all__ = ["VODPInference", "VODPRunner", "load_policy"]
