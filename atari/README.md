# STG for Atari

This repository shows how to apply STG in Atari environments.

## Requirements

```
pip -r requirements.txt
```

## Data Preparation

First and foremost, we collect a few expert observations to create expert datasets. 



## Instructions

- sac_atari.py 用SAC默认训练5000000步获得专家策略，模型和训练结果保存至checkpoints/SAC_BreakoutNoFrameskip-v4_seed0(举例)，包括events.out.tfevents.\*和SAC_epi\*_rtn\*.pth和train.csv
``` 
python sac_atari.py --env_id BreakoutNoFrameskip-v4 --clipframe 1 --total_timesteps 5000000 --mode train --seed 0  # 训练示例
python sac_atari.py --env_id BreakoutNoFrameskip-v4 --clipframe 1 --mode test --buffer_size 10000 # 在线交互采集专家轨迹，注意修改ckpts和save_dir
```

- ppo-atari.py PPO baseline，模型和训练结果保存至checkpoints/PPO_BreakoutNoFrameskip-v4_seed0(举例)，包括events.out.tfevents.\*和PPO_epi\*_rtn\*.pth和return.pkl
``` 
python ppo_atari.py --env_id BreakoutNoFrameskip-v4 --clipframe 1 --total_timesteps 10000000 --seed 0  # 训练示例
```

2.STG

- SSmodelTDC.py 训练SS模型，分类自监督预测相邻两个状态时间步差距，结果保存在ssmodel-exp\Freeway_expert_TDC_atariCNN_class5_seed666中，包括\*.pt(完整模型而不是仅模型参数)和训练数据train.csv。直接运行即可训练模型，注意修改game和src_root和Config.n_epoch。
- SSmodelTDR.py 训练SS模型，回归自监督预测相邻两个状态时间步差距，类似SSmodelTDC.py
- ppo-intrinsic-ss.py 加入intrinsic的PPO，三种intrinsic：prediction_error(pe),discrimination_score(ds),progress_span(ps)，模型和训练结果保存同上，注意ss_rewards.pkl

``` 
CUDA_VISIBLE_DEVICES=0 python ppo-intrinsic-ss.py --seed 0 --intrinsic ds --ds_coef -1 --env_reward 1 --ssmodel \*.pth  # environment_reward-discrimination_score训练示例
CUDA_VISIBLE_DEVICES=0 python ppo-intrinsic-ss.py --seed 0 --intrinsic dsps --ds_coef -1 --ps_coef 0.1 --env_reward 0 --ssmodel \*.pth  #  -discrimination_score + 0.1 * temporal_progress训练示例
```

3.OTG

- OOmodelTDC.py 训练OO模型，类似SSmodelTDC.py

- ppo-Intrinsic-oo.py 类似ppo-intrinsic-ss.py

4.comparision

- SSmodelELE.py 类似SSmodelTDC.py使用，减少了SSTransformer模块直接用progress作为intrinsic，pipeline也是先SSmodelELE.py预训练再ppo-intrinsic-ss.py
- GAIfO.py 直接在线训练Discriminator获得intrinsic，直接执行在线训练

## 
