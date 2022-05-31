"""
Script to add the image paths to each action in the game jsons. 
"""

import copy
import glob
import json
import os
import pickle
import time

import numpy as np
from tqdm import tqdm
from PIL import Image

from teach.dataset.dataset import Dataset
from teach.logger import create_logger
from teach.dataset.task_THOR import Task_THOR

def main(base_dir, phase):
    print(f"Processing {base_dir}/{phase}")
    games = []

    data_dir = os.path.join(base_dir, "games", phase)
    # images_dir = os.path.join(base_dir, "images", phase)
    images_dir = os.path.join(base_dir, "images", phase)
    new_data_dir = os.path.join(base_dir, "games_with_image_paths", phase)

    os.makedirs(new_data_dir, exist_ok=True)

    game_files = os.listdir(data_dir)

    # Go through all game files
    for game_file in tqdm(game_files):
        f = os.path.join(data_dir, game_file)
        with open(f) as h:
            game = json.load(h)
        
        new_game = copy.deepcopy(game)
        interactions = game["tasks"][0]["episodes"][0]["interactions"]

        commander_obs, driver_obs, target_obj_frame, target_obj_mask, pc_status = {}, {}, {}, {}, {}
        game_id = game_file.split(".")[0]
        game_image_dir = os.path.join(images_dir, game_id)

        # Go through all the images and add them to dictionary based on timestamp
        for img_file in os.listdir(game_image_dir):
            img_f = os.path.join(game_image_dir, img_file)
            img_f = img_f.replace(base_dir, "")[1:]
            time_start = ".".join(img_file.split(".")[2:4])
                
            tag, subtag, *_ = img_file.split(".")

            if tag == "commander":
                commander_obs[time_start] = img_f
            elif tag == "driver":
                driver_obs[time_start] = img_f
            elif tag == "targetobject":
                if subtag == "frame":
                    target_obj_frame[time_start] = img_f
                if subtag == "mask":
                    target_obj_mask[time_start] = img_f
            elif subtag == "status":
                pc_status[time_start] = img_f
            else:  ### add other metadata here
                pass
                # raise NotImplementedError(f"{tag} is not supported")

        # Iterate through all the interactions and insert the metadata paths
        for idx in range(len(interactions)):
            time_start = str(interactions[idx]["time_start"])
            interactions[idx]["commander_obs"] = commander_obs[time_start]
            interactions[idx]["driver_obs"] = driver_obs[time_start]

            # For the progress check actions
            try:
                if interactions[idx]["action_id"] == 500:
                    interactions[idx]["pc_json"] = pc_status[time_start]
            except:
                print(f)
            
            # For other commander actions
            if interactions[idx]["action_id"] > 500 and interactions[idx]["success"]:
                interactions[idx]["targetobject_frame"] = target_obj_frame[time_start]
                interactions[idx]["targetobject_mask"] = target_obj_mask[time_start]


        new_game["tasks"][0]["episodes"][0]["interactions"] = interactions
        new_f = os.path.join(new_data_dir, os.path.basename(f))

        with open(new_f, 'w') as h:
            json.dump(new_game, h)


if __name__ == "__main__":
    base_data_dir = "/home/anthony/teach_data"

    for phase in ["train", "valid_seen", "valid_unseen"]:
        main(base_data_dir, phase)
