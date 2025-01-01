# stores config for trainer, tester, and visualization files
cfg = {
        # main cfgs
        "output_dir": "output",
        "task": "square",
        "datatype": "image_abs",

        "n_components": 6,
        
        # testing cfgs
        'testing_dir': '/home/george/diffusion_policy/ood_3d/output/square/low_dim_abs/08-27-2024_17-14-47',
        # 'testing_dir': '/home/george/diffusion_policy/ood_3d/output/square/low_dim_abs/09-06-2024_13-38-08',
        'n_pts_tested': 100,

        # # rec cfg
        # "eps": -40,
        # "tau": 7.5,
        # "eta": 1.0,
        # 'obsact_ckpt': '/home/george/diffusion_policy/data/outputs/2024.07.30/12.20.59_train_diffusion_unet_lowdim_obsact_pusht_lowdim_obsact/checkpoints/epoch=3800-train_action_mse_error=207.801.ckpt',
        # 'base_ckpt': '/home/george/diffusion_policy/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/epoch=0300-train_mean_score=0.996.ckpt',
        # 'base_ckpt': '/home/george/diffusion_policy/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/latest.ckpt',
    }   

cfg["datapath"] = f'/home/george/diffusion_policy/data/robomimic/datasets/{cfg["task"]}/ph/{cfg["datatype"]}.hdf5'

combined_policy_cfg = {
    # best yet
    # 'recovery_ckpt': '/home/george/diffusion_policy/data/outputs/2024.09.05/18.57.19_train_diffusion_unet_lowdim_obsact_square_lowdim/checkpoints/latest.ckpt',
    'recovery_ckpt': '/home/george/diffusion_policy/data/outputs/2024.09.11/08.56.33_train_diffusion_unet_lowdim_obsact_square_lowdim/checkpoints/epoch=0400-train_action_mse_error=0.003.ckpt',
    # 'base_ckpt': "/home/george/diffusion_policy/data/outputs/2024.09.05/16.12.10_train_diffusion_unet_lowdim_square_lowdim_abs/checkpoints/epoch=0250-test_mean_score=0.960.ckpt",
    'base_ckpt': "/home/george/diffusion_policy/data/outputs/2024.10.09/08.07.46_train_diffusion_unet_image_square_image/checkpoints/latest.ckpt", # vision
}



