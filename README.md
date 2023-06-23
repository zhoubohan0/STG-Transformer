# State-To-Go (STG) Transformer

![](./src/main-workflow.jpg)

This is the official implementation of "Learning from Visual Observation via Offline Pretrained State-to-Go Transformer". A two-stage method, named **STG**, is proposed for reinforcement learning from visual observation.  

- In the first stage, State-to-Go (STG) Transformer is pretrained offline to predict and differentiate latent transitions of demonstrations. 
- In the second stage, the STG Transformer provides intrinsic rewards for downstream reinforcement learning tasks.

Experiments are conducted in two video game environments: Atari and Minecraft. Codes & instructions about Atari can refer to  **atari** and **minecraft** directory respectively.

## To begin
```
pip clone https://github.com/zhoubohan0/STG-Transformer
```


## Citation

Our paper is available on [Arxiv](). If you find our code useful or want to conduct further research based on STG, please consider citing us!

```bibtex
@misc{zhou2023learning,
      title={Learning from Visual Observation via Offline Pretrained State-to-Go Transformer}, 
      author={Bohan Zhou and Ke Li and Jiechuan Jiang and Zongqing Lu},
      year={2023},
      eprint={2306.12860},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Acknowledgement

We acknowledge [cleanrl](https://github.com/vwxyzjn/cleanrl) to provide a clear and simple implementation of PPO and SAC algorithm for our project. And the codebase from [Plan4MC](https://github.com/PKU-RL/Plan4MC) enables us to collect enough expert trajectories and  to tackle Minecraft tasks.

## License

MIT License
