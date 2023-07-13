# This shell is for main loop of multiple experiments.

CONFIG_FILE="./params/tide_online.ini"
DATA_FOLDER="3_Non-rigid-Real"
MODEL_FOLDER="./data/Pretrained-Models"
RUN_TAG="online-wonbr2"

for (( i=1; i<=8; i++ ))
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main.py \
        --config ${CONFIG_FILE} \
        --train_dir ./data/${DATA_FOLDER} \
        --out_dir ./output/${DATA_FOLDER} \
        --model_dir ${MODEL_FOLDER} \
        --loss_type pf \
        --run_tag ${RUN_TAG}_exp$i
done