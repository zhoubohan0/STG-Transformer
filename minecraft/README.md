# STG-Transformer for Minecraft Tasks



## Overview

The completed tasks to run STG model for MineCraft tasks are as below.

## Task id for each task

Milk a cow: `harvest_milk_with_empty_bucket_and_cow`

Gather wool: `harvest_wool_with_shears_and_sheep`

Harvest tallgrass: `harvest_1_tallgrass`

Pick a flower: `harvest_1_double_plant`


Note that for `harvest_1_tallgrass` and `harvest_1_double_plant` task, you need to modify the source code of minedojo. 

1. Open file  `MineDojo/minedojo/tasks/description_files/tasks_specs.yaml` and add following code:

    ```python
    # ====== obtain tallgrass ====
    harvest_1_tallgrass:
      __cls__: "harvest"
      prompt: "obtain a tallgrass in the plains"
      target_names: "tallgrass"
      target_quantities: 1
      reward_weights: 1
      use_voxel: true
      voxel_size: { xmin: -1, ymin: -1, zmin: -1, xmax: 1 ymax: 1, zmax: 1 }
      use_lidar: false
      initial_inventory:
        mainhand:
          name: "shears"

    # ====== obtain flower (double plant) ====
    harvest_1_double_plant:
      __cls__: "harvest"
      prompt: "obtain a sunflower in the plain"
      target_names: "double_plant"
      target_quantities: 1
      reward_weights: 1
      use_voxel: true
      voxel_size: { xmin: -1, ymin: -1, zmin: -1, xmax: 1, ymax: 1, zmax: 1 }
      use_lidar: false
      initial_inventory:
        mainhand:
          name: "shears"

    ```

2. Open file `MineDojo/minedojo/tasks/description_files/programmatic_tasks.yaml` and add following code: 
    ```python
    harvest_1_tallgrass:
      category: harvest
      guidance: '1. Find a tallgrass block.
        1. Mine the tallgrass block.
        2. Collect the dropped item.'
      prompt: obtain tallgrass

    harvest_1_double_plant:
      category: harvest
      guidance: '1. Find a double plant block.
        2. Mine the double plant block.
        3. Collect the dropped item.'
      prompt: obtain a sunflower
    ```
  
3. Open file `MineDojo/minedojo/tasks/__init__.py` and find `natural_items`. Add `tallgrass` and `double_plant` into the list.


## Step 0: Set up the MineDojo environment

- We recommend you to use anaconda to create a virtual env with python >= 3.9:
  
  ```
  conda create -n mc-env python=3.9
  ```
  
- Install MineDojo following the [official doc](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites). 
  
- Upgrade the MineDojo package to our modified version

    - Uninstall the previous version
        ```
        pip uninstall minedojo
        ```
    - Clone the modified repo
        ```
        git clone https://github.com/PKU-RL/MCEnv.git
        ```
    - Run the setup file
        ```
        cd MCEnv && python setup.py install 
        ```
  - Verify the installation 
      ```
      MINEDOJO_HEADLESS=1 python validate_install.py
      ```
      You should see "[INFO] Installation Success" if your installation is successful.
      
      Note that we add `MINEDOJO_HEADLESS=1` as a prefix to avoid Malmo error. This can happen when your system does not support visualization display. Otherwise, yuo can feel free to remove it.  
      
- Clone our STG-Transformer repo
  ```
  git clone https://github.com/zhoubohan0/STG-Transformer.git
  ```

- Install python packages in `STG-Transformer/minecraft/requirements.txt`. Note that we require PyTorch >= 1.8.1 and x-transformers==0.27.1.
  ```
  pip install -r requirements.txt 
  ```
  If the speed is too low, try this:
  ```
  pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
  ```


## Step 1: Collect expert trajectories

As we discussed in our paper, we utilized the learned policy in [Plan4MC](https://github.com/PKU-RL/Plan4MC) and [CLIP4MC](https://github.com/PKU-RL/CLIP4MC) to collect our expert datasets. Specifically, 

### Milk and Wool
- Download the [pretrained MineCLIP model](https://disk.pku.edu.cn:443/link/86843F120DF784DCC117624D2E90A569) named `attn.pth`.  Move this to the directory `minecraft/mineagent/official`.

- Enter into the the directory `STG-Transformer/minecraft`

  ```
  cd minecraft
  ```

- Run the following code to collect expert datasets for task 'milk a cow'. If you want to collect expert datasets for task 'gather wool', change the parameter `--task` to `harvest_wool_with_shears_and_sheep`

    ``` 
    python generate_expert_traj.py --task harvest_milk_with_empty_bucket_and_cow --test-episode 100
    ```
  If you encounter Malmo error, add `MINEDOJO_HEADLESS=1` at the head of this command.

  You can see the collected expert datasets under the newly created directory `minecraft/expert_traj`. '1' indicates the trajectory collected is successful while '0' indicates the trajectory is unsuccessful; 

 

### Tallgrass and Flower

For these two tasks, we trained a CLIP4MC policy from scratch and use the successful gifs saved during the training stage as our expert datasets. Please refer to [CLIP4MC](https://github.com/PKU-RL/CLIP4MC) for more details about how to train CLIP4MC policy.


## Step2: Train STG model

For each task, you should train a STG model using the expert datasets collected in Step 1. 

Run the following code:
```
python STGTrain/STG4MC.py --stg_name milk --src_dir expert_traj/harvest_milk_with_empty_bucket_and_cow --n_epoch 100
```

The `--src_dir` parameter indicates the directory where your expert datasets are saved.

After running this, you should see there is a newly created directory `stgmodel-exp`. A csv file `train.csv ` is saved to record the training loss. `*.pth` files are also saved to store the model weights.

To check whether the STG model has converged, you can visualize it via following code:

```
python plot/stg_plot.py --file stgmodel-exp/milk/trainss.csv
```

## Step 3: Train RL task

Run the following code under `STG-Transformer/minecraft` directory. Remember to replace your STG model path.

```
MINEDOJO_HEADLESS=1 python train.py --task harvest_milk_with_empty_bucket_and_cow --biome plains --ss_model_path YOUR_STG_MODEL_PATH --env_reward 0 --stg 1 --expseed 666 --horizon 500 --steps 2000 --intri_type ds --ss_coff -5 --algo stg --exp-name ppo --epochs 1000
```

Intermediate results will be saved in the `checkpoint` and `data` directory.

## Step 4: Reload model and continue training

In case of unexpected suspension, you can resume your training by simply adding a parameter `--continue-n` in your command. For example, your experiments are interrupted at epoch 654, and in the `checkpoint` the latest model is saved at 650. Then you can re-start your training by the following code:

Caution! Remember to backup the files saved in `data` directory, as it will be over-written.

```
MINEDOJO_HEADLESS=1 python train.py --task harvest_milk_with_empty_bucket_and_cow --biome plains --ss_model_path YOUR_STG_MODEL_PATH --env_reward 0 --stg 1 --expseed 666 --horizon 500 --steps 2000 --intri_type ds --ss_coff -5 --algo stg --exp-name ppo --epochs 1000 --continue-n 650
```

