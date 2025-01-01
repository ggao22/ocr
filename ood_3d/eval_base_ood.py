"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
sys.path.append('../')

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from utils import to_obj_pose, gen_keypoints, abs_traj, abs_se3_vector, deabs_se3_vector

from sklearn.mixture import GaussianMixture
from models import GMMGradient
from config import cfg as rec_cfg




def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=True,
        use_image_obs=False, 
    )
    return env


def add_obs(new_obs, past_obs, n_obs_steps):
    new_obs = new_obs[None]
    if len(past_obs) < 1:
        past_obs = np.repeat(new_obs, n_obs_steps, 0)
    else:
        old_obs = past_obs[:-1]
        past_obs = np.vstack((new_obs,old_obs))
    return past_obs


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    cfg.task.dataset['dataset_path'] = '../' + cfg.task.dataset['dataset_path']
    dataset = h5py.File(cfg.task.dataset['dataset_path'],'r')

    fig, ax = plt.subplots()

    def animate(args):
        env_img, env_num = args
        ax.cla()
        ax.imshow(env_img)
        ax.set_title(f"Env #{str(env_num)}")

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # read from dataset
    dataset_path = os.path.expanduser(cfg.task.dataset['dataset_path'])
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    
    if cfg.task.abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    env_obs_keys = [
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos']
    
    robomimic_env = create_env(
            env_meta=env_meta, 
            obs_keys=env_obs_keys,
        )
    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = RobomimicLowdimWrapper(
        env=robomimic_env,
        obs_keys=env_obs_keys,
        init_state=None,
        render_hw=(256,256),
        render_camera_name='agentview',
    )

    vec2rot6d = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')

    env_imgs = []
    max_iter = 35
    n_obs_steps = cfg.n_obs_steps
    envs_tested = list(range(1))
    ood_offsets = np.random.uniform([-0.05,-0.3],[0.05,-0.15],(len(envs_tested),2))
    env_labels = []
    rewards = []
    for n in envs_tested:
        env.init_state = dataset[f'data/demo_{n}/states'][0]
        # i=10,11,12 is xyz of object
        # env.init_state[10:12] = env.init_state[10:12] + ood_offsets[n]
        obs = env.reset()
        past_obs = []
        past_obs = add_obs(obs, past_obs, n_obs_steps)
        img = env.render(mode='rgb_array')

        # env policy control
        for _ in range(max_iter):
            np_obs_dict = {
                # handle n_latency_steps by discarding the last n_latency_steps
                'obs': past_obs[None].astype(np.float32)
            }
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action'].squeeze(0)
            action = np.hstack((action[:,:3], 
                                vec2rot6d.inverse(action[:,3:9]),
                                action[:,9:]))
            
            # step env and render
            for i in range(action.shape[0]):
                act = action[i]
                obs, reward, done, info = env.step(act)
                past_obs = add_obs(obs, past_obs, n_obs_steps)
                img = env.render(mode='rgb_array')
                env_imgs.append(img)
                env_labels.append(n)
                if reward > 0.98: done = True
                if done: break
            if done: break
        rewards.append(reward)
        if done:
            print(f'Env #{n} done')
        else:
            print(f'Env #{n} failed, max iter reached. Reward Managed {reward}')

    
    print(f"Test done, average reward {np.mean(rewards)}")
    ani = FuncAnimation(fig, animate, frames=zip(env_imgs,env_labels), interval=100, save_count=sys.maxsize)
    ani.save(os.path.join(output_dir,'base_id.mp4'), writer='ffmpeg', fps=10, dpi=400) 
    plt.show()


def generate_kp_traj(kp_start, recovery_vec, horizon, delay, alpha=0.01):
    n_kp,d_kp = kp_start.shape
    kp_base = np.repeat([kp_start], horizon, axis=0) # horizon,n_kp,D
    mean_recovery_vec = recovery_vec.mean(axis=0) * alpha
    motion_vecs = np.repeat([mean_recovery_vec], horizon-delay, axis=0) 
    motion_vecs = np.vstack((np.zeros((delay, d_kp)),motion_vecs)) # horizon,D
    vecs = np.repeat(np.cumsum(motion_vecs, axis=0), n_kp, axis=0).reshape(horizon, n_kp, d_kp)
    return kp_base + vecs


if __name__ == '__main__':
    main()
