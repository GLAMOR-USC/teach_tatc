from sacred import Ingredient
from sacred.settings import SETTINGS

seq2seq_ingredient = Ingredient("seq2seq")

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@seq2seq_ingredient.config
def cfg_seq2seq():
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

    # language embedding size
    demb = 768
    # hidden layer size
    dhid = 512
    # image feature vec size
    dframe = 2500

    # dropout rate for attention
    attn_dropout = 0
    # dropout rate for actor fc
    actor_dropout = 0
    # dropout rate for LSTM hidden states during unrolling
    hstate_dropout = 0.3
    # dropout rate for Resnet feats
    vis_dropout = 0.3
    # dropout rate for concatted input feats
    input_dropout = 0
    # dropout rate for language (goal + instr)
    lang_dropout = 0
    # use teacher forcing
    dec_teacher_forcing = True

    # zero out goal language
    zero_goal = False
    # zero out step-by-step instr language
    zero_instr = False

    save_every_epoch = 0