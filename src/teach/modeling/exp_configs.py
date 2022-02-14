from sacred import Ingredient
from sacred.settings import SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

exp_ingredient = Ingredient("exp")
eval_ingredient = Ingredient("eval")


@exp_ingredient.config
def cfg_exp():
    # HIGH-LEVEL MODEL SETTINGS
    # where to save model and/or logs
    name = "default"
    # model to use
    model = "transformer"
    # which agent is training
    agent = "commander"
    # which device to use
    device = "cuda"
    # number of data loading workers or evaluation processes (0 for main thread)
    num_workers = 12
    # we can fine-tune a pre-trained model
    pretrained_path = None
    # run the code on a small chunk of data
    fast_epoch = False

    # Set this to 1 if running on a Mac and to large numbers like 250 if running on EC2
    lmdb_max_readers = 1

    # DATA SETTINGS
    data = {
        # dataset name(s) for training and validation
        "train": None,
        # additional dataset name(s) can be specified for validation only
        "valid": "",
        # specify the length of each dataset
        "length": 30000,
        # what to use as annotations: {'lang', 'lang_frames', 'frames'}
        "ann_type": "lang",
        # Train dataloader type - sample or shuffle ("sample" results in sampling length points per epoch with
        # replacement and "shuffle" results in iterating through the train dataset in random order per epoch
        "train_load_type": "shuffle",
    }

    lang_pretrain_over_history_subgoals = False


@eval_ingredient.config
def cfg_eval():
    # which experiment to evaluate (required)
    exp = None
    # which checkpoint to load ('latest.pth', 'model_**.pth')
    checkpoint = "latest.pth"
    # which split to use ('train', 'valid_seen', 'valid_unseen')
    split = "valid_seen"
    use_sample_for_train = True
    use_random_actions = False
    no_lang = False
    no_vision = False

    # shuffle the trajectories
    shuffle = False
    # max steps before episode termination
    max_steps = 1000
    # max API execution failures before episode termination
    max_fails = 10
    # subgoals to evaluate independently, eg:all or GotoLocation,PickupObject or 0,1
    subgoals = ""
    # smooth nav actions (might be required based on training data)
    smooth_nav = False
    # forward model with expert actions (only for subgoals)
    no_model_unroll = False
    # no teacher forcing with expert (only for subgoals)
    no_teacher_force = False
    # run in the debug mode
    debug = False
    # X server number
    x_display = "0"
    # range of checkpoints to evaluate, (9, 20, 2) means epochs 9, 11, 13, 15, 17, 19
    # if None, only 'latest.pth' will be evaluated
    eval_range = (9, 20, 1)
    # object predictor path
    object_predictor = None

    eval_type = "tatc"

    # Set this to 1 if running on a Mac and to large numbers like 250 if running on EC2
    # lmdb_max_readers = 1

    # Set this to true if the model was trained (and should for inference try to get a wide view)
    wide_view = False

    force_retry = False