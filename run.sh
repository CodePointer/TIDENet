CONFIG_FILE="./params/oade_online.ini"
DATA_DIR="/media/qiao/Videos/SLDataSet/OANet"
TRAIN_FOLDER="31_VirtualData"
OUT_FOLDER="31_VirtualData-out"
MODEL_FOLDER="21_VirtualData-out/model"

for FRM_FIRST in 32 64 96 128 192
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

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config  \
#     --run_tag phpf4

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag phpf5

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag phpf6

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag phpf7

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf1

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf2

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf3

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf4

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf5

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf6

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --config ./params/oade_online.ini \
#     --run_tag pf7