# STG for Atari

This repository shows how to apply STG in Atari environments. All subsequent commands will be **in the current directory** (where the `README.md` is located).

## Requirements

- python=3.7
- atari_py=0.2.6
- gym=0.19.0
- numpy,torch,pandas,matplotlib,glob,tensorboardX,scikit-learn,gsutil

Create a conda virtual environment named "STG" and complete installation.

```
conda create -n STG python=3.7
conda activate STG
pip -r requirements.txt
```

## Data Preparation

There are two ways to form an expert dataset for STG/ELE offline pretraining in the first pretraining stage. You can download the Google [Dopamine](https://github.com/google/dopamine) Dataset using gsutil and to access to the Atari datasets just like d4rl with  [d4rl-atari](https://github.com/takuseno/d4rl-atari). 

1. Install [d4rl-atari](https://github.com/takuseno/d4rl-atari).

   ```
   pip install git+https://github.com/takuseno/d4rl-atari
   ```

2. Choose a environment and create its directory for storage.

   ```
   mkdir -p ./dataset/$env$
   ```

   `$env$` can be referred to [Atari Environment List](https://www.gymlibrary.dev/environments/atari/complete_list/), like Breakout.

3.  Run `process_data.py` to collect observation trajectories.

   ```python
   python dataset/process_Dopamine.py --game $env$
   ```

If you discover missing data or find it difficult to download the dataset, we provide another alternative to train an expert SAC agent to collect expert observations following instructions below: 

1. Train an expert SAC agent from scratch.

   ```python
   python sac_expert.py --env_id $env$NoFrameskip-v4 --mode train --buffer_size 1000000 --seed 0
   ```

   After training, the SAC checkpoints will be stored in  `sac-exp/SAC_$env$NoFrameskip-v4_seed0/SAC*.pth` and the training log will be recorded in `sac-exp/SAC_$env$NoFrameskip-v4_seed0/events.out.tfevents.*` .

2. Use the trained SAC agent to collect trajectories.

   ```python
   python sac_expert.py --env_id $env$NoFrameskip-v4 --mode test --buffer_size 100000 --test_checkpoints sac-exp/SAC_$env$NoFrameskip-v4_seed0/SAC*.pth
   ```

Regardless of the methods, all collected trajectories will be stored in `dataset/$env$`. Each `*.pkl`  save continuous expert stacked states.

## Pretraining

For example, to offline pretrain a State-to-go Transformer with collected expert observations in Breakout environment,  just run `pretrain.py`:

```python
python pretrain.py --algo STG --game Breakout --g_coff 0.3 --l2_coff 0.5 --tdr_coff 0.1
```

Similar commands for ELE:

```python
python pretrain.py --algo ELE --game Breakout --tdr_coff 0.5 
```

> If you want to use your own observation dataset or have moved the generated dataset, just additionally set `--src_root`. Be cautious that previous record will be replaced if you run this command mutiple times.

After pretraining, the pretrained model weights will be saved in `pretrain-exp/$env$_$model$/$model$_*.pth`. All statistical data of loss functions will be recorded in `pretrain-exp/$env$_$model$/$train*.csv`. We can plot the learning curve by running `vis_csv.py`:

```python
python visualize/vis_csv.py --root pretrain-exp/$env$_$model$
```

If the curve of total loss nearly converges, it is time to finish pretraining.

## RL via Pretrained Model

The pretrained model will be subsequently used for provide intrinsic rewards for downstream RL tasks. To implement STG in Breakout environment as an example, you can run `ppo_intrinsic.py` following:

```python
python ppo_intrinsic.py --algo STG --env_reward 0 --env_name BreakoutNoFrameskip-v4 --seed 0 --pretrained_model pretrain-exp/Breakout_STG/Breakout_*.pth --total_timesteps 20000000 --intrinsic ds --ds_coef -0.6 --gae_lambda 0.1
```

By modifying `algo` and `pretrained_model` , you can implement experiments of ELE/STG- with the same setting. If you want to try STG*, you can simply modify intrinsic:
```python
python ppo_intrinsic.py --algo STG --env_reward 0 --env_name BreakoutNoFrameskip-v4 --seed 0 --pretrained_model pretrain-exp/Breakout_STG/Breakout_*.pth --total_timesteps 20000000 --intrinsic dsps --ds_coef -0.6 --ps_coef 0.01 --gae_lambda 0.1
```

Also, we compare our method with online algorithm GAIfO, which has no pretraining phase in comparision with STG and ELE. You can simply run our `GAIfO.py` in Breakout as following:

```python
python ppo_GAIfO.py --algo GAIfO --env_name BreakoutNoFrameskip-v4 --seed 0 --total_timesteps 20000000 --intrinsic_coef 2.0
```

The checkpoints will be saved in `ppo-exp/*/*.pth` and the tensorboard records will be stored in `ppo-exp/*/events.out.tfevents.*`. 

## Visualization

After running `ppo_intrinsic.py` and `GAIfO.py` in each environment using each algorithm for at least 4 seeds, the training results can be naturally and clearly visualized by running: 

```python
python visualize/vis_tb.py
```

The curves with mean and standard deviation will be displayed in `vis-res/$env$.pdf`. By modifying the labels and data groups, we can also visualize more ablation results.

To check the quality of learned representation,  we use Grad-CAM for CNN visualization. 

```python
python visualize/vis_cam.py --env Breakout --algo STG
```

By running `vis_cam.py` as above, we can visualize saliency maps of different CNN layers. The same for ELE and other envvironments. The visualization results will be saved in `vis-res`.
