#!/bin/bash
export TEACH_DATA=/data/anthony/teach/tatc_preprocessed_data
export TEACH_ROOT_DIR=/data/anthony/teach_tatc
export TEACH_LOGS=/data/anthony/teach/experiments/checkpoints
export VENV_DIR=/data/anthony/envs/teach
export TEACH_SRC_DIR=$TEACH_ROOT_DIR/src

export MODEL_ROOT=$TEACH_SRC_DIR/teach/modeling
export ET_ROOT=$TEACH_SRC_DIR/teach/modeling/models/ET
export SEQ2SEQ_ROOT=$TEACH_SRC_DIR/teach/modeling/models/seq2seq_attn
export PYTHONPATH="$TEACH_SRC_DIR:$TEACH_SRC_DIR/teach:$MODEL_ROOT:$ET_ROOT:$SEQ2SEQ_ROOT:$PYTHONPATH"