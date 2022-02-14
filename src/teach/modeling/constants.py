import os

########################################################################################################################
# General Settings

MODEL_ROOT = os.environ["MODEL_ROOT"]
TEACH_DATA = os.environ["TEACH_DATA"] if "TEACH_DATA" in os.environ else None
TEACH_LOGS = os.environ["TEACH_LOGS"] if "TEACH_LOGS" in os.environ else None
TEACH_SRC = os.environ["TEACH_SRC_DIR"] if "TEACH_SRC_DIR" in os.environ else None

PAD = 0

########################################################################################################################

# TRAIN AND EVAL SETTINGS
# evaluation on multiple GPUs
NUM_EVAL_WORKERS_PER_GPU = 3
# vocabulary file name
VOCAB_FILENAME = "data.vocab"
# vocabulary with object classes
OBJ_CLS_VOCAB = "vocabs/obj_cls.vocab"

#############################

OBJECTS_ACTIONS = [
    "None",
    "AlarmClock",
    "Apple",
    "AppleSliced",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Book",
    "Bowl",
    "Box",
    "Bread",
    "BreadSliced",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Candle",
    "Cart",
    "CellPhone",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Desk",
    "DeskLamp",
    "DiningTable",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "Faucet",
    "FloorLamp",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    "LettuceSliced",
    "Microwave",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Pan",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "PotatoSliced",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "Shelf",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "TVStand",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "Tomato",
    "TomatoSliced",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
]
