# Single GPU

cd target_applications/totalsegmentator
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/root/autodl-tmp/SuPreM/target_applications/totalsegmentator/dataset/TotalSegmentator_dataset/
arch=swinunetr # support swinunetr, unet, and segresnet
suprem_path=pretrained_weights/supervised_suprem_swinunetr_2100.pth
target_task=vertebrae
num_target_class=25
num_target_annotation=64

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT \
    train.py --dist False --model_backbone $arch \
    --log_name efficiency.$arch.$target_task.number$num_target_annotation \
    --map_type $target_task --num_class $num_target_class --dataset_path $datapath \
    --num_workers 8 --batch_size 2 --pretrain $suprem_path --percent $num_target_annotation
