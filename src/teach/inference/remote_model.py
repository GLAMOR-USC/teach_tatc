# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import logging
import sys
from argparse import ArgumentParser
from io import BytesIO
from typing import List

import requests

from teach.inference.teach_model import TeachModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

TEACH_MODEL_API_URL_PREDICT = "http://{}/get_next_action"
TEACH_MODEL_API_URL_START_GAME = "http://{}/start_new_game_instance"
TEACH_MODEL_API_URL_TEST = "http://{}/test"


class RemoteModelException(Exception):
    def __init__(self, message):
        super().__init__(message)


def assign_api_by_process_idx(host_and_ports, process_index):
    splits = host_and_ports.split(",")
    if process_index >= len(splits):
        raise RemoteModelException(
            f"process_index={process_index} can't be handled by available APIs:{splits}"
        )
    return splits[process_index].strip()


class RemoteModel(TeachModel):
    def __init__(self, process_index: int, num_processes: int,
                 model_args: List[str]):

        parser = ArgumentParser()
        parser.add_argument(
            "--model_api_host_and_port",
            type=str,
            default="localhost:5000",
            help="Teach Model API hosts and ports, E.g.:api1:5000,api2:5000",
        )
        args = parser.parse_args(model_args)

        host_and_port = assign_api_by_process_idx(args.model_api_host_and_port,
                                                  process_index)
        self.test_url = TEACH_MODEL_API_URL_TEST.format(host_and_port)
        self.predict_url = TEACH_MODEL_API_URL_PREDICT.format(host_and_port)
        self.start_game_url = TEACH_MODEL_API_URL_START_GAME.format(
            host_and_port)

    def get_next_action(self,
                        img,
                        game_instance,
                        prev_action,
                        img_name=None,
                        game_name=None):
        if not img or not game_instance:
            logger.warning("either img or game_instance is None")
            return None, None
        img_in_memory = BytesIO()
        img.save(img_in_memory, "jpeg")
        img_in_memory.seek(0)
        data = {
            "img_name": img_name,
            "game_name": game_name,
            "prev_action": json.dumps(prev_action) if prev_action else None,
            "game_instance": json.dumps(game_instance),
        }

        resp = requests.post(
            self.predict_url,
            data=data,
            files={"img": (img_name, img_in_memory, "image/jpeg")})

        if resp.status_code != 200:
            logger.debug(f"failed sending data={data}")
            raise RemoteModelException(resp.text)

        resp_json = resp.json()
        action = resp_json.get("action")
        obj_relative_coord = resp_json.get("obj_relative_coord")
        return action, obj_relative_coord

    def test_connection(self):
        resp = requests.get(self.test_url)
        return resp.status_code == 200

    def start_new_game_instance(self,
                               game_instance,
                               game_name=None):
        images = []
        data = {"game_name": game_name, "game_instance": json.dumps(game_instance)}
        resp = requests.post(self.start_game_url, data=data, files=images)

        if resp.status_code != 200:
            logger.debug(f"failed sending data={data}")
            raise RemoteModelException(resp.text)

        return True
