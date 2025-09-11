#python3 inference_rsvg.py --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
#--resume rsvg_dirs/r50_bidrection_fusion_10query/checkpoint.pth --backbone resnet50 --device cpu
python3 inference_rsvg.py --dataset_file rsvg_mm --num_queries 10 --with_box_refine --binary --freeze_text_encoder \
--resume rsvg_mm_dirs/r50_bidrection_fusion_10query_70epo/checkpoint.pth --backbone resnet50 --device cpu

