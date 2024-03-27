CUDA_VISIBLE_DEVICES='2,7' python -m torch.distributed.launch --nproc_per_node=2  --master_port 29500 --use_env \
main.py --dataset_file rsvg --binary --with_box_refine \
--batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
--output_dir rsvg_dirs/r50_bidrection_fusion_10query_70epo_multiscale --backbone resnet50 \



# CUDA_VISIBLE_DEVICES='2,7' python -m torch.distributed.launch --nproc_per_node=2  --master_port 29500 --use_env \
# main.py --dataset_file rsvg_mm --binary --with_box_refine \
# --batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 \
# --output_dir rsvg_mm_dirs/r50_bidrection_fusion_10query_70epo_pretrain_multiscale --backbone resnet50 \
# --pretrained_weights rsvg_dirs/r50_bidrection_fusion_10query/checkpoint.pth

