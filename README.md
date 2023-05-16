# State-To-Go (STG) Transformer

![](C:\Users\Lenovo\Desktop\STG-Transformer\src\main-workflow.jpg)

This is the official implementation of "Learning from Visual Observation via Offline Pretrained State-to-Go Transformer". A two-stage method, named **STG**, is proposed for reinforcement learning from visual observation.  

- In the first stage, State-to-Go (STG) Transformer is pretrained offline to predict and differentiate latent transitions of demonstrations. 
- In the second stage, the STG Transformer provides intrinsic rewards for downstream reinforcement learning tasks.

Experiments are conducted in two video game environments: Atari and Minecraft. Codes & instructions about Atari can refer to  **atari** and **minecraft** directory respectively.



## Citation

Our paper is available on [Arxiv](). If you find our code useful or want to conduct further research based on STG, please consider citing us!

```bibtex
@article{
      year={2023},
}
```


## Acknowledgement

We acknowledge [cleanrl](https://github.com/vwxyzjn/cleanrl) to provide a clear and simple implementation of PPO and SAC algorithm for our project. And the codebase from [Plan4MC](https://github.com/PKU-RL/Plan4MC) enables us to collect enough expert trajectories and  to tackle Minecraft tasks.

## License

MIT License