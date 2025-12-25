from typing import Dict
import time
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from drrm.models.base_policy import BasePolicy
from drrm.models.policy.vodp.vision.obs_encoder import VODPEncoder
from drrm.models.policy.vodp.diffusion.conditional_unet1d import ConditionalUnet1D
from drrm.models.policy.vodp.diffusion.mask_generator import LowdimMaskGenerator
from drrm.models.policy.vodp.common.normalizer import LinearNormalizer
from drrm.models.policy.vodp.common.pytorch_util import dict_apply
from drrm.models.policy.vodp.common.module_attr_mixin import ModuleAttrMixin

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VODPConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: DDPMScheduler
    obs_encoder: VODPEncoder
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    num_inference_steps: int = None
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    out_channels: int = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.auto_map = {}
        self.pkg_map = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_customed_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_dict(cls, config_dict):
        return cls(**config_dict)


class VODP(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPConfig

    def __init__(self, config: VODPConfig):
        super().__init__(config)
        action_shape = config.shape_meta['action']['shape']
        noise_scheduler = hydra.utils.instantiate(config.noise_scheduler)
        obs_encoder = hydra.utils.instantiate(config.obs_encoder)
        horizon = config.horizon
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        num_inference_steps = config.num_inference_steps
        obs_as_global_cond = config.obs_as_global_cond
        diffusion_step_embed_dim = config.diffusion_step_embed_dim
        down_dims = config.down_dims
        kernel_size = config.kernel_size
        n_groups = config.n_groups
        cond_predict_scale = config.cond_predict_scale
        
        # parse shapes
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        # obs_feature_dim = obs_encoder.output_shape()[0]
        obs_feature_dim = config.out_channels

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        # self.kwargs = kwargs
        self.kwargs = {} #

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        total_start = time.time()
        
        # 初始化轨迹
        init_start = time.time()
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        print(f"[TIMING] conditional_sample.init_trajectory: {(time.time() - init_start) * 1000:.2f}ms")
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        num_steps = len(scheduler.timesteps)
        print(f"[TIMING] conditional_sample.num_steps: {num_steps}")

        # 扩散采样循环
        diffusion_start = time.time()
        model_time = 0.0
        scheduler_time = 0.0
        conditioning_time = 0.0
        
        for i, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            cond_start = time.time()
            trajectory[condition_mask] = condition_data[condition_mask]
            conditioning_time += time.time() - cond_start

            # 2. predict model output (GPU运算)
            model_start = time.time()
            if i == 0:
                # 第一次推理可能包含CUDA初始化，单独计时
                torch.cuda.synchronize() if condition_data.device.type == 'cuda' else None
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            if condition_data.device.type == 'cuda':
                torch.cuda.synchronize()  # 确保GPU计算完成
            model_time += time.time() - model_start

            # 3. compute previous image: x_t -> x_t-1
            scheduler_start = time.time()
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            scheduler_time += time.time() - scheduler_start
        
        diffusion_time = time.time() - diffusion_start
        print(f"[TIMING] conditional_sample.diffusion_loop_total: {diffusion_time * 1000:.2f}ms")
        print(f"[TIMING] conditional_sample.model_forward (GPU): {model_time * 1000:.2f}ms ({model_time/diffusion_time*100:.1f}%)")
        print(f"[TIMING] conditional_sample.scheduler_step: {scheduler_time * 1000:.2f}ms ({scheduler_time/diffusion_time*100:.1f}%)")
        print(f"[TIMING] conditional_sample.conditioning: {conditioning_time * 1000:.2f}ms ({conditioning_time/diffusion_time*100:.1f}%)")
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        
        total_time = time.time() - total_start
        print(f"[TIMING] conditional_sample.total: {total_time * 1000:.2f}ms")

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        total_start = time.time()
        
        assert 'past_action' not in obs_dict # not implemented yet
        
        # normalize input
        normalize_start = time.time()
        def key_matches(key, params_dict):
            """检查键是否在 normalizer 的参数字典中（支持点分隔键）"""
            if key in params_dict:
                return True
            # 对于点分隔键，检查前缀是否在 params_dict 中
            if '.' in key:
                prefix = key.split('.')[0]
                return prefix in params_dict
            return False
        
        filtered_obs_dict = {key: value for key, value in obs_dict.items() 
                if key_matches(key, self.normalizer.params_dict)}
        
        if len(filtered_obs_dict) == 0:
            obs_keys = list(obs_dict.keys())
            normalizer_keys = list(self.normalizer.params_dict.keys())
            raise ValueError(
                f"No matching observation keys found. "
                f"Input keys: {obs_keys}, "
                f"Normalizer expects keys starting with: {normalizer_keys}. "
                f"Please check the key mapping."
            )
        
        nobs = self.normalizer.normalize(filtered_obs_dict)
        print(f"[TIMING] predict_action.normalize: {(time.time() - normalize_start) * 1000:.2f}ms")
        
        if len(nobs) == 0:
            raise ValueError("Normalized observation dictionary is empty after normalization")
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        obs_encode_start = time.time()
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
            # 观测编码 (GPU运算)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            encoder_start = time.time()
            nobs_features = self.obs_encoder(this_nobs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print(f"[TIMING] predict_action.obs_encoder (GPU): {(time.time() - encoder_start) * 1000:.2f}ms")
            
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
            # 观测编码 (GPU运算)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            encoder_start = time.time()
            nobs_features = self.obs_encoder(this_nobs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print(f"[TIMING] predict_action.obs_encoder (GPU): {(time.time() - encoder_start) * 1000:.2f}ms")
            
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        print(f"[TIMING] predict_action.obs_encode_total: {(time.time() - obs_encode_start) * 1000:.2f}ms")

        # run sampling (主要耗时)
        sampling_start = time.time()
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        print(f"[TIMING] predict_action.conditional_sample_total: {(time.time() - sampling_start) * 1000:.2f}ms")
        
        # unnormalize prediction
        unnormalize_start = time.time()
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        print(f"[TIMING] predict_action.unnormalize: {(time.time() - unnormalize_start) * 1000:.2f}ms")

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        total_time = time.time() - total_start
        print(f"[TIMING] predict_action.total: {total_time * 1000:.2f}ms")
        print(f"[TIMING] predict_action.device: {device}")
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss