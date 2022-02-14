# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
from argparse import ArgumentParser
from os.path import isfile

from flask import Flask, jsonify, request
from flask_restful import reqparse
from PIL import Image

from teach.utils import dynamically_load_class, load_images

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
app.logger.info("initialize flask server")


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help='Base data directory containing subfolders "games" and "games',
    )
    arg_parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Images directory containing inference image output",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=[
            "train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"
        ],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )
    arg_parser.add_argument(
        "--model_module",
        type=str,
        default="teach.inference.sample_model",
        help="Path of the python module to load the model class from.",
    )
    arg_parser.add_argument(
        "--model_class",
        type=str,
        default="SampleModel",
        help="Name of the TeachModel class to use during inference.")
    arg_parser.add_argument("--use_game_file",
                            dest="use_game_file",
                            action="store_true",
                            help="Use game file instead of request json.")
    arg_parser.add_argument("--use_img_file",
                            dest="use_img_file",
                            action="store_true",
                            help="Use img file instead of request bytes.")
    return arg_parser.parse_known_args()


teach_args, model_args = parse_args()
model_class = dynamically_load_class(teach_args.model_module,
                                     teach_args.model_class)
process_index, num_processes = 1, 1
model = model_class(process_index, num_processes, model_args=model_args)


def _get_game_instance(req_args):
    if teach_args.use_game_file:
        if not req_args.game_name:
            return None, "request parameter game_name does not have a value"
        game_instance_path = os.path.join(teach_args.data_dir, "games",
                                          teach_args.split, req_args.game_name)
        if not isfile(game_instance_path):
            return None, f"game file={game_instance_path} does not exist"
        with open(game_instance_path) as handle:
            game_instance = json.load(handle)
    else:
        game_instance = json.loads(req_args.game_instance)
    return game_instance, None


def _get_img(req_args):
    if not req_args.img_name:
        return None, "request parameter img_name does not have a value"
    if teach_args.use_img_file:
        img_path = os.path.join(teach_args.images_dir, req_args.img_name)
        if not isfile(img_path):
            return None, f"image file={img_path} does not exist"
        img = Image.open(img_path)
    else:
        img_file = request.files.get("img")
        if not img_file:
            return None, f"image is not set in request with key='img'"
        img = Image.open(img_file)
    return img, None

@app.route("/get_next_action", methods=["POST"])
def get_next_action():
    req_args = get_next_action_parse_args()
    game_instance, err_msg = _get_game_instance(req_args)
    if err_msg:
        return err_msg, 500
    img, err_msg = _get_img(req_args)
    if err_msg:
        return err_msg, 500
    prev_action = json.loads(
        req_args.prev_action) if req_args.prev_action else None
    try:
        action, obj_relative_coord = model.get_next_action(
            img, game_instance, prev_action)
    except Exception as e:
        err_msg = f"failed to get_next_action with game_name={req_args.game_name}"
        app.logger.error(err_msg, exc_info=True)
        return err_msg, 500
    app.logger.debug(
        f"model.get_next_action returns action={action}, obj_relative_coord={obj_relative_coord}"
    )
    resp = jsonify(action=action, obj_relative_coord=obj_relative_coord)
    return resp, 200


@app.route("/start_new_game_instance", methods=["POST"])
def start_new_game_instance():
    req_args = start_new_game_instance_parse_args()
    app.logger.info(
        f"start_new_game_instance with game_name={req_args.game_name}")
    game_instance, err_msg = _get_game_instance(req_args)
    if err_msg:
        return err_msg, 500
    try:
        model.start_new_game_instance(game_instance)
    except Exception as e:
        err_msg = f"failed to start_new_game_instance with game_name={req_args.game_name}"
        app.logger.error(err_msg, exc_info=True)
        return err_msg, 500
    return "success", 200


@app.route("/")
@app.route("/ping")
@app.route("/test")
def test():
    resp = jsonify(action="Look Up", obj_relative_coord=[0.1, 0.2])
    return resp, 200


def get_next_action_parse_args():
    parser = reqparse.RequestParser()
    parser.add_argument(
        "img_name",
        type=str,
        help="Image name for PIL Image containing agent's egocentric image.",
    )
    parser.add_argument(
        "game_name",
        type=str,
        help="game instance file name.",
    )
    parser.add_argument(
        "prev_action",
        type=str,
        help=
        "One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values.",
    )
    parser.add_argument(
        "game_instance",
        type=str,
        help=
        "One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values.",
    )
    args = parser.parse_args()
    return args


def start_new_game_instance_parse_args():
    parser = reqparse.RequestParser()
    parser.add_argument(
        "game_name",
        type=str,
        help="game instance file name.",
    )
    parser.add_argument(
        "game_instance",
        type=str,
        help=
        "One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values.",
    )
    args = parser.parse_args()
    return args


def main():
    app.run(host="0.0.0.0", port=5000)
    app.logger.info("started flask server")


if __name__ == "__main__":
    main()
