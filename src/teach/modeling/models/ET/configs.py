from sacred import Ingredient
from sacred.settings import SETTINGS

et_ingredient = Ingredient("et")

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@et_ingredient.config
def cfg_et():
    # GENERAL TRANING SETTINGS
    # random seed
    seed = 1
    # load a checkpoint from a previous epoch (if available)
    resume = True
    # whether to print execution time for different parts of the code
    profile = False

    # For ablations
    no_lang = False
    no_vision = False

    # HYPER PARAMETERS
    # batch size
    batch = 8
    # number of epochs
    epochs = 20
    # optimizer type, must be in ('adam', 'adamw')
    optimizer = "adamw"
    # L2 regularization weight
    weight_decay = 0.33
    # learning rate settings
    lr = {
        # learning rate initial value
        "init": 1e-4,
        # lr scheduler type: {'linear', 'cosine', 'triangular', 'triangular2'}
        "profile": "linear",
        # (LINEAR PROFILE) num epoch to adjust learning rate
        "decay_epoch": 10,
        # (LINEAR PROFILE) scaling multiplier at each milestone
        "decay_scale": 0.1,
        # (COSINE & TRIANGULAR PROFILE) learning rate final value
        "final": 1e-5,
        # (TRIANGULAR PROFILE) period of the cycle to increase the learning rate
        "cycle_epoch_up": 0,
        # (TRIANGULAR PROFILE) period of the cycle to decrease the learning rate
        "cycle_epoch_down": 0,
        # warm up period length in epochs
        "warmup_epoch": 0,
        # initial learning rate will be divided by this value
        "warmup_scale": 1,
    }
    # weight of action loss
    action_loss_wt = 1.0
    # weight of object loss
    action_aux_loss_wt = 1.0
    # weight of subgoal completion predictor
    subgoal_aux_loss_wt = 0
    # weight of progress monitor
    progress_aux_loss_wt = 0
    # maximizing entropy loss (by default it is off)
    entropy_wt = 0.0

    # Should train loss be computed over history actions? (default False)
    compute_train_loss_over_history = False

    # TRANSFORMER settings
    # size of transformer embeddings
    demb = 768
    # number of heads in multi-head attention
    encoder_heads = 12
    # number of layers in transformer encoder
    encoder_layers = 2
    # how many previous actions to use as input
    num_input_actions = 1
    # which encoder to use for language encoder (by default no encoder)
    encoder_lang = {
        "shared": True,
        "layers": 2,
        "pos_enc": True,
        "instr_enc": False,
    }
    # which decoder to use for the speaker model
    decoder_lang = {
        "layers": 2,
        "heads": 12,
        "demb": 768,
        "dropout": 0.1,
        "pos_enc": True,
    }

    # do not propagate gradients to the look-up table and the language encoder
    detach_lang_emb = False

    # DROPOUTS
    dropout = {
        # dropout rate for language (goal + instr)
        "lang": 0.0,
        # dropout rate for Resnet feats
        "vis": 0.3,
        # dropout rate for processed lang and visual embeddings
        "emb": 0.0,
        # transformer model specific dropouts
        "transformer": {
            # dropout for transformer encoder
            "encoder": 0.1,
            # remove previous actions
            "action": 0.0,
        },
    }

    # ENCODINGS
    enc = {
        # use positional encoding
        "pos": True,
        # use learned positional encoding
        "pos_learn": False,
        # use learned token ([WORD] or [IMG]) encoding
        "token": False,
        # dataset id learned encoding
        "dataset": False,
    }