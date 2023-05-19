import minedojo
import numpy as np

from mineagent.official import torch_normalize
from mineagent.batch import Batch
import torch
from mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP

def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        ret = np.zeros(3*3*3, dtype=np.long)
        for i, v in enumerate(vox.reshape(3*3*3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        if act[5]<=1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5]==3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            #raise Exception('Action[5] should be 0,1,3')
            return 0

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    obs_ = {
        "compass": torch.as_tensor([np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)])], device=device),
        "gps": torch.as_tensor([obs["location_stats"]["pos"]], device=device),
        "voxels": torch.as_tensor(
            [cvt_voxels(obs["voxels"]["block_name"])], dtype=torch.long, device=device
        ),
        "biome_id": torch.tensor(
            [int(obs["location_stats"]["biome_id"])], dtype=torch.long, device=device
        ),
        "prev_action": torch.tensor(
            [cvt_action(obs["prev_action"])], dtype=torch.long, device=device
        ),
        "prompt": torch.as_tensor(obs["rgb_emb"], device=device).view(B, 512), 
        # this is actually the image embedding, not prompt embedding (for single task)
        #"goal": torch.as_tensor(obs["goal_emb"], dtype=torch.float, device=device).view(B, 6), 
    }
    #print(obs_["prev_action"])
    #print(obs_["compass"], yaw_, pitch_)
    #print(obs_["goal"])

    #print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map agent action to env action.

# for [3,3,4,25,25,8] agent action space
'''
def transform_action(action):

    assert action.ndim == 2 # (1, 6)
    action = action[0]
    action = action.cpu().numpy()
    if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
        action[-1] = 0
    action = np.concatenate([action, np.array([0, 0])])
    return action #(8)
'''

# [56, 3] agent action space as I initially implemented
'''
def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 56
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 < 20: # camera pitch 0~11
        action[3] = act1 - 8
    elif act1 < 32: # camera pitch 13~24
        action[3] = act1 - 8 + 1
    elif act1 < 44: # camera yaw 0~11
        action[4] = act1 - 32
    else: # camera yaw 13~24
        action[4] = act1 - 32 + 1

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)
'''

# for [3,3,4,5,3] action space, preserve only 5 camera choices 
'''
def transform_action(act):
    assert act.ndim == 2 # (1, 5)
    act = act[0]
    act = act.cpu().numpy()
    
    action = [act[0],act[1],act[2],12,12,0,0,0] #self.base_env.action_space.no_op()

    # no_op, use, attack
    act_use = act[4]
    if act_use == 2:
        act_use = 3
    action[5] = act_use

    # no_op, 2 pitch, 2 yaw
    act_cam = act[3]
    if act_cam == 1:
        action[3] = 11
    elif act_cam == 2:
        action[3] = 13
    elif act_cam == 3:
        action[4] = 11
    elif act_cam == 4:
        action[4] = 13

    #print(action)

    return action #(8)
'''

# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 5 camera actions

def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10
        action[3] = 10
    elif act1 == 9: # camera pitch 14
        action[3] = 14
    elif act1 == 10: # camera yaw 10
        action[4] = 10
    elif act1 == 11: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)


'''
9/6 
To support dense reward, you should insert these codes
    for key in kwargs:
        if key in task_specs:
            task_specs.pop(key)
into your MineDojo package minedojo/tasks/_init__.py line 494, before calling '_meta_task_make'

'''
from collections import deque
class MinecraftEnv:

    def __init__(self, task_id, image_size=(160, 256), max_step=500, clip_model=None, device=None, seed=0,
        dense_reward=False, target_name=None,  biome=None, dis=5, ):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.dense_reward = dense_reward
        self.dis = dis
        self.biome = biome

        if not dense_reward:
            self.base_env = minedojo.make(task_id=task_id, image_size=image_size, seed=seed, 
            initial_mob_spawn_range_low=[-self.dis,1,-self.dis], initial_mob_spawn_range_high=[self.dis,1,self.dis], 
            specified_biome=self.biome,)
        else:
            self.base_env = minedojo.make(task_id=task_id, image_size=image_size, seed=seed, 
                initial_mob_spawn_range_low=[-self.dis,1,-self.dis], initial_mob_spawn_range_high=[self.dis,1,self.dis], 
                use_lidar=True, lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 50)
                    for pitch in np.arange(-30, 30, 6)
                    for yaw in np.arange(-45, 45, 9)])
            self._target_name = target_name
            self._consecutive_distances = deque(maxlen=2)
            self._distance_min = np.inf
            self._weapon_durability_deque = deque(maxlen=2)
        self.max_step = max_step
        self.cur_step = 0
        self.task_prompt = self.base_env.task_prompt
        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.device = device
        self.seed = seed
        self.task_id = task_id
        self.image_size = image_size

        self._first_use_weapon_durability_deque = True
        self._first_reset = True
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    def remake_env(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
            import deepcopy
            task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        self.base_env.close()
        if not self.dense_reward:
            self.base_env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed, 
            initial_mob_spawn_range_low=[-self.dis,1,-self.dis], initial_mob_spawn_range_high=[self.dis,1,self.dis],
            specified_biome=self.biome,)
        else:
            self.base_env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed, 
            initial_mob_spawn_range_low=[-self.dis,1,-self.dis], initial_mob_spawn_range_high=[self.dis,1,self.dis],
            specified_biome=self.biome,
                use_lidar=True, lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 50)
                    for pitch in np.arange(-30, 30, 6)
                    for yaw in np.arange(-45, 45, 9)])
            self._consecutive_distances = deque(maxlen=2)
            self._distance_min = np.inf
        self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')


    def reset(self):
        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.base_env.unwrapped.execute_cmd(cmd)
            self.base_env.unwrapped.set_time(6000)
            self.base_env.unwrapped.set_weather("clear")
        self._first_reset = False
        self.prev_action = self.base_env.action_space.no_op()

        obs = self.base_env.reset()
        self.cur_step = 0

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        if self.dense_reward:
            self._consecutive_distances.clear()
            self._weapon_durability_deque.clear()  # updated 11.29
            self._first_use_weapon_durability_deque = True  # updated 11.29


        return obs

    def step(self, act):
        obs, reward, done, info = self.base_env.step(act)
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True
        
        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        self.prev_action = act # save the previous action for the agent's observation

        if self.dense_reward:
            valid_attack = 0
            if (self._first_use_weapon_durability_deque == True):
                self._first_use_weapon_durability_deque = False
                self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])
                self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])
                valid_attack = (
                        self._weapon_durability_deque[0] - self._weapon_durability_deque[1]
                )


                valid_attack = 1.0 if valid_attack == 1.0 else 0.0
            else:
                # self._weapon_durability_deque.pop()    # In Python, elements in the deque move forward in turn when a new element appended
                self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])
                valid_attack = (
                        self._weapon_durability_deque[0] - self._weapon_durability_deque[1]
                )
                # when dying, the weapon is gone and durability changes to 0
                valid_attack = 1.0 if valid_attack == 1.0 else 0.0

            if (valid_attack == 1.0):
                for i in range(obs["rays"]["ray_yaw"].size):
                    if obs["rays"]["ray_yaw"][i] == 0 and obs["rays"]["ray_pitch"][i] == 0 and \
                            obs["rays"]["entity_name"][i] == self._target_name:
                        reward = 1.0
                        done = True
                        print("attack!")
                        break
            # no dense reward       
            obs['dense_reward'] = 0
                    

        return  obs, reward, done, info
