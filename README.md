# State-To-Go (STG) Transformer

This is the official implementation of "Learning from Visual Observation via Offline Pretrained State-to-Go Transformer". A two-stage framework, named **STG**, is proposed for reinforcement learning from visual observation.  

- In the first stage, State-to-Go (STG) Transformer is offline to predict and differentiate latent transitions of demonstrations. 
- In the second stage, the STG Transformer provides intrinsic rewards for downstream reinforcement learning tasks .

Experiments are conducted in two video game environments: **Atari** and **Minecraft**. Codes and instructions are listed in respective directory.



## Citation

Our paper is available on [Arxiv](). If you find our code useful or want to conduct further research based on STG, please consider citing us!


## Acknowledgement

We acknowledge [cleanrl](http://jmlr.org/papers/v23/21-1342.html) to provide a clear and simple implementation of PPO and SAC algorithm for our project.

## License

MIT 