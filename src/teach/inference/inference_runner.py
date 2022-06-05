# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import json
import multiprocessing as mp
import os
import pdb
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import isdir
from pathlib import Path
from typing import List, Type

from PIL import Image

from teach.dataset.definitions import Definitions
from teach.dataset.interaction import Interaction
from teach.dataset.task import Task
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
    get_state_changes,
)
from teach.dataset.task_THOR import Task_THOR

definitions = Definitions(version="2.0")
action_id_to_info = definitions.map_actions_id2info
logger = create_logger(__name__)


@dataclass
class InferenceRunnerConfig:
    data_dir: str
    split: str
    output_dir: str
    images_dir: str
    model_class: Type[TeachModel]
    model_args: List[str]
    metrics_file: str = "metrics.json"
    num_processes: int = 1
    max_init_tries: int = 3
    max_traj_steps: int = 1000
    max_api_fails: int = 30
    use_img_file: bool = False
    replay_timeout: int = 500


class InferenceRunner:
    def __init__(self, game_files, config: InferenceRunnerConfig):
        self._game_files = game_files
        self._config = config

    def run(self):
        self._launch_processes(self._game_files, self._config)
        return self._load_metrics()

    def _load_metrics(self):
        metrics = dict()
        for metrics_file in InferenceRunner._get_metrics_files(self._config):
            if os.path.isfile(metrics_file):
                with open(metrics_file) as h:
                    thread_replay_status = json.load(h)
                metrics.update(thread_replay_status)
        return metrics

    @staticmethod
    def _get_metrics_files(config):
        return [
            InferenceRunner._get_metrics_file_name_for_process(
                x, config.metrics_file) for x in range(config.num_processes)
        ]

    @staticmethod
    def _launch_processes(game_files, config: InferenceRunnerConfig):
        processes = []
        ers = []
        try:
            for process_index in range(config.num_processes):
                er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
                ers.append(er)
                process = InferenceRunner._launch_process(
                    process_index, game_files, config, er)
                processes.append(process)
        finally:
            InferenceRunner._join_processes(processes)
            for er in ers:
                er.simulator.shutdown_simulator()

    @staticmethod
    def _launch_process(process_index, game_files,
                        config: InferenceRunnerConfig, er: EpisodeReplay):
        num_files = len(game_files)
        num_files_per_process = InferenceRunner._get_num_files_per_process(
            num_files=num_files, num_processes=config.num_processes)


        start_index, end_index = InferenceRunner._get_range_to_process(
            process_index=process_index,
            num_files_per_process=num_files_per_process,
            num_files=num_files,
        )

        files_to_process = game_files[start_index:end_index]

        InferenceRunner._run(process_index, files_to_process, config, er)
        return None

        # process = mp.Process(target=InferenceRunner._run, args=(process_index, files_to_process, config, er))

        # process.start()
        # time.sleep(0.1)
        # return process

    @staticmethod
    def _run(process_index, files_to_process, config: InferenceRunnerConfig,
             er: EpisodeReplay):
        metrics_file = InferenceRunner._get_metrics_file_name_for_process(
            process_index, config.metrics_file)
        metrics = dict()

        model = config.model_class(process_index,
                                   config.num_processes,
                                   model_args=config.model_args)

        for file_index, instance_file in enumerate(files_to_process):
            try:
                instance_id, instance_metrics = InferenceRunner._run_game(
                    instance_file, config, model, er)
                metrics[instance_id] = instance_metrics
                save_dict_as_json(metrics, metrics_file)

                logger.info(
                    f"Instance {instance_id}, metrics: {instance_metrics}")
                logger.info(
                    f"Process {process_index} completed {file_index + 1} / {len(files_to_process)} instances"
                )
            except Exception:
                err_msg = f"exception happened for instance={instance_file}, continue with the rest"
                logger.error(err_msg, exc_info=True)
                continue

    @staticmethod
    def _run_game(instance_file, config: InferenceRunnerConfig,
                  model: TeachModel, er: EpisodeReplay):
        instance_id = copy.deepcopy(instance_file)
        instance_id = instance_id.replace(config.data_dir + '/', "")
        instance_id = instance_id.replace(config.split + '/', "")
        instance_id = instance_id.split('.')[0]

        game = InferenceRunner._load_game(instance_file)
        game['instance_id'] = instance_id

        game_file = InferenceRunner._get_game_file(game, config)

        metrics = create_new_traj_metrics(game)
        logger.debug(f"Processing instance {instance_id}")

        er.set_episode_by_fn_and_idx(game_file, 0, 0)

        api_success, init_state = er._set_up_new_episode(None, turn_on_lights=False)

        (
            success,
            initial_goal_conditions_total,
            initial_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(er,
                                                    er.simulator.current_task)

        ## if success==1: there's nothing to do in this episode
        assert initial_goal_conditions_satisfied <= initial_goal_conditions_total

        model_started_success = False
        try:
            model_started_success = model.start_new_tatc_instance(game)
        except Exception:
            model_started_success = False
            metrics["error"] = 1
            logger.error(
                f"Failed to start_new_tatc_instance for {instance_id}",
                exc_info=True)

        if model_started_success:

            # Keep track of history
            driver_action_history, commander_action_history = [], []
            driver_pose_history, commander_pose_history = [], []
            dialogue_history = [("", "commander")]
            dialogue = []
            commander_action_result_history = []

            er.simulator.is_record_mode = True
            pred_actions = list()

            traj_steps_taken = 0
            for _ in range(config.max_traj_steps):
                traj_steps_taken += 1
                try:
                    commander_img, driver_img = InferenceRunner._get_latest_image(er)
                    commander_pose, driver_pose = InferenceRunner._get_poses(er)

                    commander_img_name = InferenceRunner._save_image(
                        config, "commander", game, commander_img, traj_steps_taken)
                    driver_img_name = InferenceRunner._save_image(
                        config, "driver", game, driver_img, traj_steps_taken)

                    # Get next commander action
                    commander_inputs = dict(
                        driver_img=driver_img,
                        driver_position=driver_pose,
                        commander_img=commander_img,
                        commander_curr_pose=commander_pose,
                        commander_action_history=commander_action_history,
                        commander_pose_history=commander_pose_history,
                        driver_action_history=driver_action_history,
                        driver_pose_history=driver_pose_history,
                        driver_obs_history=None,
                        dialogue_history=dialogue_history,
                        commander_action_result_history=commander_action_result_history
                    )
                    commander_action = model.get_next_action_commander(commander_inputs, game, commander_img_name, instance_file)

                    # Get next driver action
                    driver_inputs = dict(
                        driver_img=driver_img,
                        driver_curr_pose=driver_pose,
                        dialogue_history=dialogue_history,
                        driver_action_history=driver_action_history,
                        driver_pose_history=driver_pose_history,
                        driver_obs_history=None
                    )
                    driver_action = model.get_next_action_driver(driver_inputs, game, driver_img_name, instance_file)

                    # Execute actions in simulator
                    commander_step_success, result = InferenceRunner._execute_commander_action(er.simulator, **commander_action)

                    # Save progress check result if commander executes a PC action
                    if commander_action['action'] == "OpenProgressCheck":
                        commander_action_result_history.append(result)

                    driver_step_success = InferenceRunner._execute_driver_action(er.simulator, **driver_action)
                    InferenceRunner._update_metrics(metrics, 
                                                    commander_action,
                                                    driver_action,
                                                    commander_step_success,
                                                    driver_step_success)

                    driver_action['success'] = driver_step_success
                    commander_action['success'] = commander_step_success
                    driver_action_history.append(driver_action)
                    commander_action_history.append(commander_action)
                    driver_pose_history.append(driver_pose)
                    commander_pose_history.append(commander_pose)

                    pred_actions.append([commander_action, driver_action])

                    # if commander_action['utterance'] and driver_action['utterance']:
                    #     import ipdb; ipdb.set_trace()
                    
                    if commander_action['utterance']:
                        dialogue_history.append((commander_action['utterance'], "commander"))

                    if driver_action['utterance']:
                        dialogue_history.append((driver_action['utterance'], "driver"))

                except Exception as e:
                    logger.error(
                        f"_run_game Exception: {str(e)} for instance_id={instance_id}, "
                        f"traj_steps_taken={traj_steps_taken}",
                        exc_info=True,
                    )
                    metrics["error"] = 1
                    break
                if InferenceRunner._should_end_inference(
                        driver_action, metrics, config.max_api_fails):
                    break

        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(er,
                                                    er.simulator.current_task)

        assert final_goal_conditions_total == initial_goal_conditions_total

        metrics_diff = evaluate_traj(
            success,
            game,
            traj_steps_taken,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
            initial_goal_conditions_satisfied
        )
        metrics.update(metrics_diff)

        os.makedirs(config.output_dir, exist_ok=True)
        pred_actions_file = os.path.join(config.output_dir, "pred_actions__" + instance_id + ".json")
        with open(pred_actions_file, "w") as handle:
            json.dump(pred_actions, handle)

        er.simulator.dir_out = config.output_dir
        output_file = os.path.join(config.output_dir,"inference__" + instance_id + ".json")
        er.simulator.save(file_name=output_file)

        return instance_id, metrics

    @staticmethod
    def _check_episode_progress(er, task):
        (
            _,
            success,
            _,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = er.simulator.check_episode_progress(task)
        return success, final_goal_conditions_total, final_goal_conditions_satisfied

    @staticmethod
    def _initialize_episode_replay(game, game_file, task, replay_timeout,
                                   er: EpisodeReplay):
        start_time = time.perf_counter()
        er.set_episode_by_fn_and_idx(game_file, 0, 0)
        interactions = list()
        ep_interactions = game["tasks"][0]["episodes"][0]["interactions"] 
        for interaction in ep_interactions:
            action = action_id_to_info[interaction["action_id"]]
            interactions.append(
                Interaction.from_dict(interaction, action["action_type"]))
        er.episode.interactions = interactions

        init_success = False
        with ThreadPoolExecutor() as tp:
            future = tp.submit(er.play_episode,
                               task=task,
                               shutdown_on_finish=False)
            logger.info(
                f"Started episode replay with timeout: {replay_timeout} sec")
            init_success, _ = future.result(timeout=replay_timeout)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Elapsed time for episode replay: {elapsed_time}")

        return init_success, er if init_success else None

    @staticmethod
    def _get_poses(er):
        return er.simulator.get_current_pose(agent_id=0), er.simulator.get_current_pose(agent_id=1)

    @staticmethod
    def _get_latest_image(er):
        images = er.simulator.get_latest_images()
        return Image.fromarray(images["allo"]), Image.fromarray(images["ego"])

    @staticmethod
    def _execute_commander_action(simulator, action, obj_cls, utterance=None):
        step_success = True
        r = None

        if action in ["OpenProgressCheck", "SearchObject", "SelectOid"]:
            r = simulator.apply_progress_check(action,
                                               agent_id=0,
                                               query=obj_cls)
        elif action in ["Text"]: #, "Speech"]:
            simulator.keyboard(agent_id=0, utterance=utterance)
        elif action == "NoOp":
            step_success = True
        else:
            step_success, _, _ = simulator.apply_motion(motion_name = action, agent_id=0)
                
        return step_success, r

    @staticmethod
    def _execute_driver_action(simulator, action, predicted_click, utterance=None):
        if action in ["NoOp"]:
            return True
        
        if action in ["Stop"]:
            return True

        if action in ["Text"]: #, "Speech"]:
            simulator.keyboard(agent_id=1, utterance=utterance)
            return True

        if action in obj_interaction_actions:
            y = predicted_click[0]
            x = predicted_click[1]
            step_success, _, _ = simulator.apply_object_interaction(
                action, 1, x, y)
            return step_success

        step_success, _, _ = simulator.apply_motion(motion_name = action, agent_id=1)
        return step_success

    @staticmethod
    def _get_game_file(game, config: InferenceRunnerConfig):
        return os.path.join(
            config.data_dir,
            config.split,
            f"{game['instance_id']}.game.json",
        )

    @staticmethod
    def _get_metrics_file_name_for_process(process_index, metrics_file):
        return f"{metrics_file}.json.{process_index}"

    @staticmethod
    def _update_metrics(metrics, commander_action, driver_action,
                        commander_step_success, driver_step_success):
        metrics["pred_actions"].append((commander_action, driver_action))

        if driver_action['action'] == "Stop":
            metrics["predicted_stop"] = 1

        if not commander_step_success or not driver_step_success:
            metrics["num_api_fails"] += 1

    @staticmethod
    def _should_end_inference(action, metrics, max_api_fails):
        return action == "Stop" or metrics["num_api_fails"] >= max_api_fails

    @staticmethod
    def _load_game(instance_file):
        with open(instance_file) as handle:
            game = json.load(handle)
        return game

    @staticmethod
    def _get_range_to_process(process_index, num_files_per_process, num_files):
        start_index = process_index * num_files_per_process
        end_index = min(start_index + num_files_per_process, num_files)
        return start_index, end_index

    @staticmethod
    def _get_num_files_per_process(num_files, num_processes):
        return int(num_files / num_processes) + 1

    @staticmethod
    def _join_processes(processes):
        for process in processes:
            process.join()

    @staticmethod
    def _save_image(config, agent, game, img, traj_steps_taken):
        image_name = f"{agent}_img__{game['instance_id']}_{traj_steps_taken}.jpeg"
        if config.use_img_file:
            InferenceRunner._save_image_sync(img, image_name, config)
        else:
            InferenceRunner._save_image_async(img, image_name, config)
        return image_name

    @staticmethod
    def _save_image_async(img, image_name, config: InferenceRunnerConfig):
        process = mp.Process(target=InferenceRunner._save_image_sync,
                             args=(img, image_name, config))
        process.start()
        return image_name

    @staticmethod
    def _save_image_sync(img, image_name, config: InferenceRunnerConfig):
        if not isdir(config.images_dir):
            Path(config.images_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(config.images_dir, image_name)
        img.save(image_path)
        return image_name
