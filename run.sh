export agent=driver # or commander
export preprocessed_path=/data/anthony/teach/tatc_preprocessed_data
CUDA_VISIBLE_DEVICES=3 python -m modeling.train \
    with exp.model=seq2seq_attn \
    exp.name=seq2seq_attn_${agent} \
    exp.data.train=${preprocessed_path} \
    exp.agent=${agent} \
    seq2seq.epochs=20 \
    seq2seq.batch=8 \
    seq2seq.seed=2 \
    seq2seq.resume=False

## change models!!
# export exp=testing
# export INFERENCE_OUTPUT_PATH=/data/ishika/teach_new/teach_tatc/experiments/${exp}
# export TEACH_LOGS=/data/anthony/teach/experiments/checkpoints
CUDA_VISIBLE_DEVICES=0  python src/teach/cli/inference.py \
    --model_module teach.inference.seq2seq_model \
    --model_class Seq2SeqModel \
    --data_dir $TEACH_DATA/games \
    --images_dir $INFERENCE_OUTPUT_PATH/images \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_tatc \
    --split valid_unseen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_tatc.json \
    --seed 0 \
    --commander_model_dir $TEACH_LOGS/seq2seq_attn_commander_final \
    --driver_model_dir $TEACH_LOGS/seq2seq_attn_driver_final \
    --visual_checkpoint $TEACH_DATA/experiments/checkpoints/pretrained/maskrcnn_model.pth \
    --device cuda \
    --preprocessed_data_dir ${preprocessed_path} 

        # --object_predictor $TEACH_LOGS/pretrained/maskrcnn_model.pth \
#experiments/exp2/inference__teach_tatc \

# echo "start"
# python src/teach/cli/replay.py \
#     --game_dir $TEACH_DATA/games/train \
#     --write_frames_dir experiments/testing/replay \
#     --write_frames \
#     --num_processes 1 \
#     --status_out_fn experiments/testing/replay/file.json
    # --create_video


# Command for generating the dataset 
CUDA_VISIBLE_DEVICES=3 python -m modeling.datasets.create_dataset \
    with args.visual_checkpoint=$TEACH_LOGS/pretrained/fasterrcnn_model.pth \
    args.data_input=$TEACH_DATA/games \
    args.data_output=test \
    args.fast_epoch=True \
    args.vocab_path=None \
    args.num_workers=0