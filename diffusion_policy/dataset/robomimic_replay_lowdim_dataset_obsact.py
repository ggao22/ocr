from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from ood_3d.utils import to_obj_pose, gen_keypoints, abs_traj, abs_se3_vector, deabs_se3_vector

class RobomimicReplayLowdimObsactDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            copy=True,
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)
                replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            copy=copy)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        datadict = _get_full_obsact_dict(self.sampler)
        stat = array_to_stats(datadict['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # aggregate obs stats
        obs_stat = array_to_stats(datadict['obs'].astype('float32'))

        normalizer['obs'] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        data = _absolute_data(data)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
   
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

    # HARD CODING
    # Setting object obs array length from 14 to 9, to fit the keypoint array length
    obs = obs[:,:9]
    

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    data = {
        'obs': obs,
        'action': raw_actions
    }
    return data


def _get_full_obsact_dict(sampler):
    datadict = {
        'obs': np.array([]),
        'action': np.array([]),
    }
    for idx in range(0,len(sampler),2):
        data = sampler.sample_sequence(idx)
        data = _absolute_data(data)
        datadict['obs'] = np.vstack((datadict['obs'], data['obs'])) if datadict['obs'].size else data['obs']
        datadict['action'] = np.vstack((datadict['action'], data['action'])) if datadict['action'].size else data['action']
    return datadict

def _absolute_data(data):
    obj_pose = to_obj_pose(data['obs']) # H,4,4
    kp = gen_keypoints(obj_pose) # H,n_kp,D_kp
    abs_kp = abs_traj(kp, obj_pose[0])
    data['obs'] = abs_kp.reshape(-1,9)
    # data['obs'] = kp.reshape(-1,9)
    
    data['action'][...,:9] = abs_se3_vector(data['action'][...,:9], obj_pose[0])
    # data['action'][...,:9] = deabs_se3_vector(data['action'][...,:9], obj_pose[0])
    return data

