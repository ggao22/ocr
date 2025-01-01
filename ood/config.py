# stores config for trainer, tester, and visualization files
cfg = {
        # main cfgs
        'loss_type': 'action, position',
        "output_dir": "output",
        "input_size": (96,96),
        "dataname": "pusht_demo_left_kp",
        "action_dim": 18,
        "space_dim": 2,

        "num_workers": 2,
        "batch_size": 64,

        "num_test_traj": 2,
        "n_components": 4,
        
        "test_every": 100,

        # testing cfgs
        'ood_datapath': '/home/george/diffusion_policy/data/pusht_demo_right_test.zarr',
        'testing_dir': '/home/george/diffusion_policy/ood/output/pusht_demo_left_kp/10-11-2024_00-22-22/',

        # rec cfg
        'obsact_ckpt': '/home/george/diffusion_policy/data/outputs/2024.07.30/12.20.59_train_diffusion_unet_lowdim_obsact_pusht_lowdim_obsact/checkpoints/epoch=3800-train_action_mse_error=207.801.ckpt',
        # 'base_ckpt': '/home/georgegao/diffusion_policy_ood/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/epoch=0300-train_mean_score=0.996.ckpt',
        'base_ckpt': '/home/george/diffusion_policy/data/outputs/2024.08.01/10.00.59_train_diffusion_unet_image_pusht_image/checkpoints/latest.ckpt',
    }   

cfg["datapath"] = f'/home/george/diffusion_policy/data/{cfg["dataname"]}.zarr'


