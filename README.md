# TEACh Two-Agent Task Completion (TATC) Challenge
[Task-driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)

Aishwarya Padmakumar*, Jesse Thomason*, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, Dilek Hakkani-Tur

TEACH (Task-drive Embodied Agents that Chat) is a set of episodes in which a human Commander who knows what task needs to be accomplished and where objects in a scene are located works with a human Follower who controls a virtual agent in the scene to accomplish household chores. The Commander and Follower communicate via natural language text. TEACh enables researchers to study strategies for natural language cooperation, instruction following from egocentric vision, question and answer generation, and partner modeling. We believe these skills will be necessary for real-world language-driven robotics applications.

The code and model weights are licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).
Please include appropriate licensing and attribution when using our data and code, and please cite our paper.

Note: The TEACh repository at https://github.com/alexa/teach is centered around the EDH and TFD challenges in which the agent must learn to produce actions from only dialogue history. This repository is adapted for the TATC challenge. In TATC, two agents *Commander* and *Follower* collaborate via natural language dialogue to solve the tasks. 

## Prerequisites
- python3 `>=3.7,<=3.8`
- python3.x-dev, example: `sudo apt install python3.8-dev`
- tmux, example: `sudo apt install tmux`
- xorg, example: `sudo apt install xorg openbox`
- ffmpeg, example: `sudo apt install ffmpeg`

## Installation
```
pip install -r requirements.txt
pip install -e .
```
## Downloading the dataset
Run the following script:
```
sh download_data.sh
```

Download and extract the archive files (`games.tar.gz`, `meta_data.tar.gz`) in the default 
directory (`/tmp/teach-dataset`). To download the data used for the EDH challenge please refer to the README at https://github.com/alexa/teach.

## Remote Server Setup
If running on a remote server without a display, the following setup will be needed to run episode replay, model inference of any model training that invokes the simulator (student forcing / RL). 

Start an X-server 
```
tmux
sudo python ./bin/startx.py
```
Exit the `tmux` session (`CTRL+B, D`). Any other commands should be run in the main terminal / different sessions. 


## Replaying episodes
Most users should not need to do this since we provide this output in `meta_data.tar.gz`.

The following steps can be used to read a `.json` file of a gameplay session, play it in the AI2-THOR simulator, and at each time step save egocentric observations of the `Commander` and `Driver` (`Follower` in the paper). It also saves the target object panel and mask seen by the `Commander`, and the difference between current and initial state.     

Replaying a single episode locally, or in a new `tmux` session / main terminal of remote headless server:
```
teach_replay \
--game_fn /path/to/game/file \
--write_frames_dir /path/to/desired/output/images/dir \
--write_frames \
--write_states \
--status-out-fn /path/to/desired/output/status/file.json
```
Note that `--status-out-fn` must end in `.json`
Also note that the script will by default not replay sessions for which an output subdirectory already exists under `--write-frames-dir`
Additionally, if the file passed to `--status-out-fn` already exists, the script will try to resume files not marked as replayed in that file. It will error out if there is a mismatch between the status file and output directories on which sessions have been previously played. 
It is recommended to use a new `--write-frames-dir` and new `--status-out-fn` for additional runs that are not intended to resume from a previous one.

Replay all episodes in a folder locally, or in a new `tmux` session / main terminal of remote headless server:
```
teach_replay \
--game_dir /path/to/dir/containing/.game.json/files \
--write_frames_dir /path/to/desired/output/images/dir \
--write_frames \
--write_states \
--num_processes 50 \
--status-out-fn /path/to/desired/output/status/file.json
```

To generate a video, additionally specify `--create_video`. Note that for images to be saved, `--write_images` must be specified and `--write-frames-dir` must be provided. For state changes to be saved, `--write_states` must be specified and `--write_frames_dir` must be provided.

# Training 

We provide implementation for a baseline seq2seq attention model from the ALFRED repository adapted for the TATC benchmark. Note that we have removed files not used when running seq2seq on TEACh, and many files have been significantly modified.

Below are instructions for training and evaluating commander and driver seq2seq models. If running on a laptop, it might be desirable to mimic the folder structure of the TEACh dataset, but using only a small number of games from each split, and their corresponding images and EDH instances. 

Set some useful environment variables. Optionally, you can copy these export statements over to a bash script and source it before training. 
```buildoutcfg
export TEACH_DATA=/tmp/teach-dataset
export TEACH_ROOT_DIR=/path/to/teach/repo
export TEACH_LOGS=/path/to/store/checkpoints
export VENV_DIR=/path/to/folder/to/store/venv
export TEACH_SRC_DIR=$TEACH_ROOT_DIR/src/teach
export INFERENCE_OUTPUT_PATH=/path/to/store/inference/execution/files
export MODEL_ROOT=$TEACH_SRC_DIR/modeling
export ET_ROOT=$TEACH_SRC_DIR/modeling/models/ET
export SEQ2SEQ_ROOT=$TEACH_SRC_DIR/modeling/models/seq2seq_attn
export PYTHONPATH="$TEACH_SRC_DIR:$MODEL_ROOT:$ET_ROOT:$SEQ2SEQ_ROOT:$PYTHONPATH"
```
Create a virtual environment

```buildoutcfg
python3 -m venv $VENV_DIR/teach_env
source $VENV_DIR/teach_env/bin/activate
cd TEACH_ROOT_DIR
pip install --upgrade pip 
pip install -r requirements.txt
```

Download the ET pretrained checkpoint for Faster RCNN and Mask RCNN models
```buildoutcfg
wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
unzip et_checkpoints.zip
mv pretrained $TEACH_LOGS/
rm et_checkpoints.zip
```

Perform data preprocessing (this extracts image features and does some processing of game jsons). 
This step is **optional** as we already provide the preprocessed version of the dataset. However, we provide the command here for those who want to perform additional preprocessing. 
```buildoutcfg
python -m modeling.datasets.create_dataset \
    with args.visual_checkpoint=$TEACH_LOGS/pretrained/fasterrcnn_model.pth \
    args.data_input=games \
    args.task_type=game \
    args.data_output=tatc_dataset \
    args.vocab_path=None
```

Note: If running on laptop on a small subset of the data, use `args.vocab_path=$MODEL_ROOT/vocab/human.vocab` and add `args.device=cpu`.


Train commander and driver models (adjust the `train.epochs` value in this command to specify the number of desired train epochs).
Also see `modeling/exp_configs.py` and `modeling/models/seq2seq_attn/configs.py` for additional training parameters. You can also run `python -m modeling.train -h` to list out the parameters and their usage. 

```buildoutcfg
agent=driver or commander
CUDA_VISIBLE_DEVICES=0 python -m modeling.train \
    with exp.model=seq2seq_attn \
    exp.name=seq2seq_attn_${agent} \
    exp.data.train=tatc_dataset \
    exp.agent=${agent} \
    seq2seq.epochs=20 \
    seq2seq.batch=8 \
    seq2seq.seed=2 \
    seq2seq.resume=False
```
Note: If running on laptop on a small subset of the data, add `exp.device=cpu` and `exp.num_workers=1`

Copy certain necessary files to the model folder so that we do not have to access training info at inference time.
```buildoutcfg
cp $TEACH_DATA/tatc_dataset/data.vocab $TEACH_LOGS/seq2seq_attn_commander
cp $TEACH_DATA/tatc_dataset/params.json $TEACH_LOGS/seq2seq_attn_commander
cp $TEACH_DATA/tatc_dataset/data.vocab $TEACH_LOGS/seq2seq_attn_driver
cp $TEACH_DATA/tatc_dataset/params.json $TEACH_LOGS/seq2seq_attn_driver
```

Evaluate the trained model
```buildoutcfg
cd $TEACH_ROOT_DIR
python src/teach/cli/inference.py \
    --model_module teach.inference.seq2seq_model \
    --model_class Seq2SeqModel \
    --data_dir $TEACH_DATA \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_tatc.json \
    --split valid_seen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_tatc.json \
    --seed 0 \
    --commander_model_dir $TEACH_LOGS/seq2seq_attn_commander \
    --driver_model_dir $TEACH_LOGS/seq2seq_attn_driver \
    --object_predictor $TEACH_LOGS/pretrained/maskrcnn_model.pth \
    --device cpu
```

## Note about model I/O

### Driver
---
Driver actions:
- Navigation, Interaction, Text

Driver input: 
- Driver current frame 
- Driver pose
- Driver action history 
- Driver observation history
- Dialogue history

Driver output:
- Navigation or interaction action
- (x, y) coordinates of object or object class selection

### Commander
--- 

Commander actions:
- OpenProgressCheck, SearchObject, SearchOid, Text

Commander inputs:
- Commander current frame
- Commander pose
- Commander action history
- Driver pose history
- Driver observation history
- Dialogue history
- Action dependent
    - OpenProgressCheck: progress check output
    - SearchObject: target mask and target object frame
    - SearchOid: target mask and target object frame

Note: Commander does not get the task as input. It needs to first call 
OpenProgressCheck to see the task and communicate the task description to the Driver. Additionally, the Commander does not have direct access to the Driver's action history. It needs to infer the Driver's action success through the Driver's visual observation (e.g. if the frame doesn't change between timesteps).

Commander outputs: 
- Action / Text
- Action dependent:
    - SearchObject: Object id 
    - SearchOid: Object name



## FAQ

## Submission
Coming soon!

<!---
We include sample scripts for inference and calculation of metrics. `teach_inference` and `teach_eval`. 
`teach_inference` is a wrapper that implements loading a game instance, interacting with the simulator as well as writing the game
file and predicted action sequence as JSON files after each inference run. It dynamically loads the model based on the `--model_module`
and `--model_class` arguments. Your model has to implement `teach.inference.teach_model.TeachModel`. See `teach.inference.sample_model.SampleModel`
for an example implementation which takes random actions at every time step. 

After running `teach_inference`, you use `teach_eval` to compute the metrics based output data produced by `teach_inference`.


Sample run:
```
export DATA_DIR=/path/to/data/with/games/as/subdirs (Default in Downloading is /tmp/teach-dataset)
export OUTPUT_DIR=/path/to/output/folder/for/split
export METRICS_FILE=/path/to/output/metrics/file_without_extension

teach_inference \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --split valid_seen \
    --metrics_file $METRICS_FILE \
    --model_module teach.inference.sample_model \
    --model_class SampleModel

teach_eval \
    --data_dir $DATA_DIR \
    --inference_output_dir $OUTPUT_DIR \
    --split valid_seen \
    --metrics_file $METRICS_FILE
```    
-->

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

The code is licensed under the MIT License (see SOFTWARELICENSE), images are licensed under Apache 2.0 
(see IMAGESLICENSE) and other data files are licensed under CDLA-Sharing 1.0 (see DATALICENSE).