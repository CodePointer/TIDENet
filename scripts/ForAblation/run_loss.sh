CONFIG_FILE="./params/oade_online.ini"
DATA_DIR="/media/qiao/Videos/SLDataSet/OANet"
# TRAIN_FOLDER="52_RealData"
# OUT_FOLDER="52_RealData-out"
TRAIN_FOLDER="31_VirtualDataEval"
OUT_FOLDER="31_VirtualDataEval-out"
MODEL_FOLDER="21_VirtualData-out/model"

for LOSS_TYPE in phpfwom  # phpfwm pf ph phpfwom
do
    for (( i=1; i<=8; i++ ))
    do
        CUDA_VISIBLE_DEVICES=${1} python -m torch.distributed.launch --nproc_per_node=1 main.py \
            --config ${CONFIG_FILE} \
            --train_dir ${DATA_DIR}/${TRAIN_FOLDER} \
            --out_dir ${DATA_DIR}/${OUT_FOLDER} \
            --model_dir ${DATA_DIR}/${MODEL_FOLDER} \
            --loss_type ${LOSS_TYPE} \
            --frm_first 0 \
            --run_tag ${LOSS_TYPE}_exp$i
    done
done