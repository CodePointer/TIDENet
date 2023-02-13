CONFIG_FILE="./params/oade_online.ini"
DATA_DIR="/media/qiao/Videos/SLDataSet/OANet"
TRAIN_FOLDER="52_RealData"
OUT_FOLDER="52_RealData-out-abs"
# TRAIN_FOLDER="31_VirtualData"
# OUT_FOLDER="31_VirtualData-out"
MODEL_FOLDER="21_VirtualData-out/model"

for FRM_FIRST in 0 32 64 96 128 160 192 224
do
    for (( i=5; i<=8; i++ ))
    do
        CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 main.py \
            --config ${CONFIG_FILE} \
            --train_dir ${DATA_DIR}/${TRAIN_FOLDER} \
            --out_dir ${DATA_DIR}/${OUT_FOLDER} \
            --model_dir ${DATA_DIR}/${MODEL_FOLDER} \
            --frm_first ${FRM_FIRST} \
            --run_tag frm${FRM_FIRST}_exp$i
    done
done