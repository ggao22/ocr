import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

    
# evaluating
def eval_encoder(imgs, encoder, device):
    with torch.no_grad():
        preprocess = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        latent_data = []
        for i in range(len(imgs)):
            latent_data.append(encoder(preprocess(imgs[i]).unsqueeze(0).to(device)).detach().cpu().numpy())
        latent_data = torch.from_numpy(np.vstack((latent_data)))
        return latent_data


# plotting
def draw_latent(zs, save_path):
    fig = plt.figure(figsize=(18,12))
    fig.tight_layout()
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o)
        x = zs[:,2*(o-1)]
        y = zs[:,2*(o-1)+1]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], color=plt.cm.rainbow(i/zs.shape[0]), s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"Point {o}")
    plt.savefig(save_path)



from matplotlib.patches import Ellipse

def draw_ood_latent(zs, ood_zs, save_path, grad_arrows=np.array(None), MVNs=[]):
    fig = plt.figure(figsize=(18,12))
    fig.tight_layout()
    for o in range(1,7):
        ax = fig.add_subplot(2, 3, o)
        x = zs[:,2*(o-1)]
        y = zs[:,2*(o-1)+1]
        oodx = ood_zs[:,2*(o-1)]
        oody = ood_zs[:,2*(o-1)+1]
        ax.scatter(x, y, color='tab:blue', s=30)
        ax.scatter(oodx, oody, color='tab:red', s=30)
        if grad_arrows.any():
            gx = grad_arrows[:,2*(o-1)]
            gy = grad_arrows[:,2*(o-1)+1]
            gu = grad_arrows[:,2*(o-1)+18]
            gv = grad_arrows[:,2*(o-1)+1+18]
            ax.quiver(gx, gy, gu, gv, angles='xy', scale_units='xy', scale=5, alpha=0.6)
        if MVNs:
            for mean, cov in zip(MVNs[o-1][0], MVNs[o-1][1]):
                v, w = np.linalg.eigh(cov)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v) 
                v = v*200

                ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color='tab:pink')
                ell.set_alpha(0.3)
                ax.add_patch(ell)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Point {o}")
    plt.savefig(save_path)



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata

def unnormalize_data(ndata, stats):
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def unnormalize_gradient(ndata, stats):
    data = ndata * (stats['max'] - stats['min'])
    return data


# centering kp
def get_center_pos(kp):
    center_pos = kp.mean(axis=0)
    return center_pos

def get_center_ang(kp):
    pt = kp[3]
    center_pos = get_center_pos(kp)
    centered_pt = pt - center_pos
    center_ang = np.arctan2(centered_pt[1],centered_pt[0])
    return center_ang

def centralize(traj, pos, ang, screen_size):
    assert traj.shape[-1] == 2

    # center pos
    traj = traj - pos

    # center angle
    correction_ang = -ang - np.pi/2
    rot_mat = np.array([
        [np.cos(correction_ang),-np.sin(correction_ang)],
        [np.sin(correction_ang),np.cos(correction_ang)]
    ])
    for i in range(len(traj)):
        pts = traj[i]
        if len(pts.shape)==2:
            pts = np.swapaxes(pts,0,1)
            pts = rot_mat @ pts
            traj[i] = np.swapaxes(pts,0,1)
        else:
            traj[i] = rot_mat @ pts
            
    traj = traj + np.array([screen_size//2, screen_size//2])
    return traj

def centralize_grad(traj, ang):
    assert traj.shape[-1] == 2

    # center angle
    correction_ang = -ang - np.pi/2
    rot_mat = np.array([
        [np.cos(correction_ang),-np.sin(correction_ang)],
        [np.sin(correction_ang),np.cos(correction_ang)]
    ])
    for i in range(len(traj)):
        pts = traj[i]
        if len(pts.shape)==2:
            pts = np.swapaxes(pts,0,1)
            pts = rot_mat @ pts
            traj[i] = np.swapaxes(pts,0,1)
        else:
            traj[i] = rot_mat @ pts
            
    return traj



def decentralize(traj, pos, ang, screen_size):
    # inverse of centralize #
    traj = traj - np.array([screen_size//2, screen_size//2])

    # decenter angle
    correction_ang = -(-ang - np.pi/2)
    rot_mat = np.array([
        [np.cos(correction_ang),-np.sin(correction_ang)],
        [np.sin(correction_ang),np.cos(correction_ang)]
    ])
    for i in range(len(traj)):
        pts = traj[i]
        if len(pts.shape)==2:
            pts = np.swapaxes(pts,0,1)
            pts = rot_mat @ pts
            traj[i] = np.swapaxes(pts,0,1)
        else:
            traj[i] = rot_mat @ pts
    
    # decenter pos
    traj = traj + pos
    return traj

    