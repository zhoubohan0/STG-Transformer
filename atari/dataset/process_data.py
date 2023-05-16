import os
import gym
import argparse
import d4rl_atari
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True)
    args = parser.parse_args()

    game = args.game
    env_id = f'{game.lower()}-epoch-50-v0'

    env = gym.make(f'{env_id}', stack=True)  # -v{0, 1, 2, 3, 4} for datasets with the other random seeds
    env.reset() # observation.shape == (4, 84, 84)
    dataset = env.get_dataset()

    # get trajectories
    index = np.nonzero(dataset['terminals'])[0]
    cnt = []
    for i,(b,e) in enumerate(zip(index[:-1][::-1],index[1:][::-1])):
        traj = np.stack(dataset['observations'][b:e])
        cnt.append(traj.shape[0])
        with open(os.path.join(os.getcwd(),game,f'epi{i}.pkl'), 'wb') as f:
            pickle.dump({'states':traj}, f)
        if sum(cnt) > 100000:
            break
    print(f"Episode counts:", len(cnt))
