# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from abc import abstractmethod
from typing import List


class TeachModel:
    @abstractmethod
    def __init__(self, process_index: int, num_processes: int,
                 model_args: List[str]):
        """
        A model will be initialized for each evaluation process.

        See sample_model.py for a sample implementation.

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        
    @abstractmethod
    def get_next_action_commander(self,
                                  commander_img,
                                  driver_img,
                                  tatc_instance,
                                  prev_action,
                                  commander_img_name=None,
                                  driver_img_name=None,
                                  tatc_name=None):
        """
        Returns a commander action
        :param commander_img: PIL Image containing commander's egocentric image
        :param driver_img: PIL Image containing driver's egocentric image
        :param tatc_instance: tatc instance
        :param prev_action: One of None or a dict with keys 'commander_action', 'obj_cls', 'driver_action', and 'obj_relative_coord' 
        containing returned values from a previous call of get_next_action
        :param commander_img_name: commander image file name
        :param driver_img_name: driver image file name
        :param tatc_name: tatc instance file name
        :return action: An action name from commander actions
        :return obj_cls: Object class for search object or select oid action
        """

    @abstractmethod
    def get_next_action_driver(self,
                               commander_img,
                               driver_img,
                               tatc_instance,
                               prev_action,
                               commander_img_name=None,
                               driver_img_name=None,
                               tatc_name=None):
        """
        Returns a driver action
        :param commander_img: PIL Image containing commander's egocentric image
        :param driver_img: PIL Image containing driver's egocentric image
        :param tatc_instance: tatc instance
        :param prev_action: One of None or a dict with keys 'commander_action', 'obj_cls', 'driver_action', and 'obj_relative_coord' 
        containing returned values from a previous call of get_next_action
        :param commander_img_name: commander image file name
        :param driver_img_name: driver image file name
        :param tatc_name: tatc instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """

    @abstractmethod
    def start_new_tatc_instance(self,
                               tatc_instance,
                               game_name=None):
        """
        This method will be called at the start of each TATC instance after the environment has been set to the
        initial state by replaying history actions but before any actions are requested from the model by calling
        get_next_action
        :param tatc_instance: TATC instance
        :param tatc_name: TATC instance file name
        """
