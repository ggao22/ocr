import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from robosuite.utils.transform_utils import quat2mat, euler2mat, mat2quat


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    matrix = torch.tensor(matrix)

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out).numpy()

def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    d6 = torch.tensor(d6)

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2).numpy()


def matrix_to_rotation_6d(matrix):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    matrix = torch.tensor(matrix)
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,)).numpy()


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = torch.tensor(quaternions)

    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3)).numpy()


# plotting
azim = 50
elev = 15

def draw_latent(zs, save_path):
    fig = plt.figure(figsize=(18,10))
    fig.tight_layout()
    for o in range(1,4):
        ax = fig.add_subplot(1, 3, o, projection='3d')
        ax.azim = azim
        ax.elev = elev
        x = zs[:,3*(o-1)+0]
        y = zs[:,3*(o-1)+1]
        z = zs[:,3*(o-1)+2]
        for i in range(len(x)):
            ax.scatter(x[i:i+1], y[i:i+1], z[i:i+1], color=plt.cm.rainbow(i/zs.shape[0]), s=30)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f"Point {o}")
    plt.savefig(save_path)


def draw_ood_latent(zs, ood_zs, save_path, grad_arrows=np.array(None), MVNs=[]):
    fig = plt.figure(figsize=(18,10))
    fig.tight_layout()
    for o in range(1,4):
        ax = fig.add_subplot(1, 3, o, projection='3d')
        ax.azim = azim
        ax.elev = elev
        x = zs[:,3*(o-1)+0]
        y = zs[:,3*(o-1)+1]
        z = zs[:,3*(o-1)+2]
        ood_x = ood_zs[:,3*(o-1)+0]
        ood_y = ood_zs[:,3*(o-1)+1]
        ood_z = ood_zs[:,3*(o-1)+2]
        ax.scatter(x, y, z, color='tab:blue', s=30)
        ax.scatter(ood_x, ood_y, ood_z, color='tab:red', s=30)
        if grad_arrows.any():
            gx = grad_arrows[:,3*(o-1)]
            gy = grad_arrows[:,3*(o-1)+1]
            gz = grad_arrows[:,3*(o-1)+2]

            gu = grad_arrows[:,3*(o-1)+9+0]
            gv = grad_arrows[:,3*(o-1)+9+1]
            gm = grad_arrows[:,3*(o-1)+9+2]
            ax.quiver(gx, gy, gz, gu, gv, gm, length=0.0005, color='k', alpha=0.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Point {o}")
    plt.savefig(save_path)



# centering kp, assuming pose are available
def abs_traj(traj, pose_0):
    assert traj.shape[-1] == 3

    ### METHOD 1 ###
    # R = scipy.spatial.transform.Rotation
    # r = R.from_matrix(pose_0[:3,:3])
    # r = r.inv()
    # inv_rot = r.as_matrix()
    # inv_pos = -pose_0[:3,3]

    # for i in range(traj.shape[0]):
    #     kps = traj[i] #n_kp,D_kp
    #     traj[i] = (inv_rot @ (kps + inv_pos).T).T

    ### METHOD 2 ###
    org_shape = traj.shape
    traj = traj.reshape(-1,3) #n_kp,D_kp

    # switch into homogenous coordinates
    homo_coord = np.ones((traj.shape[0],1))
    traj_homo = np.hstack((traj,homo_coord)) #n_kp,D_kp+1

    # centering traj on first pose
    transformed_kps_homo = np.linalg.solve(pose_0, traj_homo.T).T #n_kp,D_kp+1
    traj = transformed_kps_homo[:,:3] #n_kp,D_kp
    traj = traj.reshape(org_shape)

    return traj


def deabs_traj(traj, pose_0):
    assert traj.shape[-1] == 3
    org_shape = traj.shape
    traj = traj.reshape(-1,3) #n_kp,D_kp

    # switch into homogenous coordinates
    homo_coord = np.ones((traj.shape[0],1))
    traj_homo = np.hstack((traj,homo_coord)) #n_kp,D_kp+1

    # decentering traj on first pose
    transformed_kps_homo = (pose_0 @ traj_homo.T).T #n_kp,D_kp+1
    traj = transformed_kps_homo[:,:3] #n_kp,D_kp
    traj = traj.reshape(org_shape)
    return traj


def abs_se3_vector(se3_vector, pose_0):
    # assuming D is position + rotation_6d
    assert se3_vector.shape[-1] == 9
    
    org_shape = se3_vector.shape
    se3_vector = se3_vector.reshape(-1,org_shape[-1])

    position = se3_vector[:,:3] 
    rot = se3_vector[:,3:] 
    rot = rotation_6d_to_matrix(rot)

    position = np.expand_dims(position,2) # N,3,1
    partial_pose = np.concatenate((rot,position),2) # N,3,4
    full_pose = np.concatenate((partial_pose, np.repeat([[[0,0,0,1]]],partial_pose.shape[0],0)),1) # N,4,4
    
    # centering traj on first pose
    transformed_full_pose = np.linalg.solve(pose_0, full_pose)
    transformed_position = transformed_full_pose[:,:3,3]
    transformed_r_matrix = transformed_full_pose[:,:3,:3]

    transformed_rot = matrix_to_rotation_6d(transformed_r_matrix)
    
    transformed_se3_vector = np.hstack((transformed_position, transformed_rot))
    transformed_se3_vector = transformed_se3_vector.reshape(org_shape)
    return transformed_se3_vector


def deabs_se3_vector(se3_vector, pose_0):
    # assuming D is position + rotation_6d
    assert se3_vector.shape[-1] == 9
    
    org_shape = se3_vector.shape
    se3_vector = se3_vector.reshape(-1,org_shape[-1])

    position = se3_vector[:,:3] 
    rot = se3_vector[:,3:] 
    rot = rotation_6d_to_matrix(rot)

    position = np.expand_dims(position,2) # N,3,1
    partial_pose = np.concatenate((rot,position),2) # N,3,4
    full_pose = np.concatenate((partial_pose, np.repeat([[[0,0,0,1]]],partial_pose.shape[0],0)),1) # N,4,4
    
    # centering traj on first pose
    transformed_full_pose = pose_0 @ full_pose
    transformed_position = transformed_full_pose[:,:3,3]
    transformed_r_matrix = transformed_full_pose[:,:3,:3]

    transformed_rot = matrix_to_rotation_6d(transformed_r_matrix)
    
    transformed_se3_vector = np.hstack((transformed_position, transformed_rot))
    transformed_se3_vector = transformed_se3_vector.reshape(org_shape)
    return transformed_se3_vector


def abs_grad(traj, pose_0):
    assert traj.shape[-1] == 3
    org_shape = traj.shape
    traj = traj.reshape(-1,3).T
    pose_0_inv = np.linalg.inv(pose_0)

    # center grad on first pose
    R = pose_0_inv[:3,:3]
    traj = R @ traj
    traj = traj.reshape(org_shape)
    return traj


def robosuite_data_to_obj_dataset(data):
    # data/demo_0/obs/object
    object_dataset = []
    for demo in data['data'].keys():
        obj_obs = np.array(data['data'][demo]['obs']['object'])
        object_dataset.append(obj_obs)
    object_dataset = np.vstack((object_dataset)) # N,D
    return object_dataset

def to_obj_pose(object_dataset):
    inds = np.array([3, 0, 1, 2])
    object_rotation = quaternion_to_matrix(object_dataset[:,3:7][:,inds]) # obj dim: [nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat]
    object_pos = object_dataset[:,:3]

    object_pose = np.zeros((object_dataset.shape[0],4,4))
    object_pose[:,:3,:3] = object_rotation
    object_pose[:,:3,3] = object_pos
    object_pose[:,3,3] = 1
    
    return object_pose

def obs_quat_to_rot6d(quat, mode='robomimic'):
    if mode=='robomimic':
        mat_rotated = quat2mat(quat) @ euler2mat(np.array([0, 0, -np.pi/2]))
    elif mode=='real':
        mat_rotated = quat2mat(quat)
    else:
        raise Exception
    rot6d_corrected = matrix_to_rotation_6d(mat_rotated[None])[0]
    return rot6d_corrected

def quat_correction(quat):
    mat_rotated = quat2mat(quat) @ euler2mat(np.array([0, 0, -np.pi/2]))
    quat_corrected = mat2quat(mat_rotated)
    return quat_corrected


# kp generation from pose
def gen_keypoints(poses, est_obj_size=0.05):
    keypoints = []
    for pose in poses:
        t = pose[:3,3]
        R = pose[:3,:3]

        kp_t = []
        for i in range(3):
            kp = R[:3,i] * est_obj_size + t
            kp_t.append(kp)
        keypoints.append(kp_t)

    return np.array(keypoints)



# data stats utils
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats