class ELEConfig:
    """ base GPT config, params common to all GPT versions """
    # model hyperparameters
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    block_size = 128
    n_layer = 3
    n_head = 4
    n_embd = 512
    max_timestep = 10000
    # training hyperparameters
    weight_decay = 0.1
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    # loss coefficients
    tdr_coff = 0.5
    # training setting
    n_epoch = 200
    batch_size = 16
    device = "cuda"
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
class STGConfig:
    """ base GPT config, params common to all GPT versions """
    # model hyperparameters
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    block_size = 128
    n_layer = 3
    n_head = 4
    n_embd = 512
    max_timestep = 10000
    # training hyperparameters
    weight_decay = 0.1
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    # loss coefficients
    l2_coff = 0.5
    g_coff = 0.05
    d_coff = 0.5
    tdr_coff = 0.1
    # training setting
    n_epoch = 200
    batch_size = 16
    device = "cuda"
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
