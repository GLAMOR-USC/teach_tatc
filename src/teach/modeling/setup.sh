#!/bin/bash

export TEACH_DATA="/data/anthony/teach"
export TEACH_ROOT_DIR="/home/anthony/teach"
export TEACH_LOGS="/data/anthony/teach/experiments/checkpoints"
export VENV_DIR="/data/anthony/envs/teach"
export TEACH_SRC_DIR="$TEACH_ROOT_DIR/src/teach"
export INFERENCE_OUTPUT_PATH="/data/anthony/teach/experiments"
export MODEL_ROOT="$TEACH_SRC_DIR/modeling"
export ET_ROOT="$TEACH_SRC_DIR/modeling/models/ET"
export SEQ2SEQ_ROOT="$TEACH_SRC_DIR/modeling/models/seq2seq_attn"
export PYTHONPATH="$TEACH_SRC_DIR:$MODEL_ROOT:$ET_ROOT:$SEQ2SEQ_ROOT:$PYTHONPATH"