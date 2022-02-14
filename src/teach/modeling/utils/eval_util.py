import logging

from modeling.models.ET.modules.enc_visual import FeatureExtractor
from modeling.utils import model_util

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


def load_agent(model_name, model_path, dataset_info, args, test_mode=False):
    """
    load a pretrained agent and its feature extractor
    """
    logger.info("In load_agent, model_path = %s, dataset_info = %s" %
                (str(model_path), str(dataset_info)))
    learned_model, _ = model_util.load_model(model_name,
                                             model_path,
                                             args.device,
                                             test_mode=test_mode)

    if model_name != "seq2seq_attn":
        model = learned_model.model
    else:
        model = learned_model

    model.eval()
    model.args.device = args.device
    extractor = FeatureExtractor(
        archi=dataset_info["visual_archi"],
        device=args.device,
        checkpoint=args.visual_checkpoint,
        compress_type=dataset_info["compress_type"],
    )
    return model, extractor


def load_object_predictor(args):
    if args.object_predictor is None:
        return None
    return FeatureExtractor(
        archi="maskrcnn",
        device=args.device,
        checkpoint=args.object_predictor,
        load_heads=True,
    )
