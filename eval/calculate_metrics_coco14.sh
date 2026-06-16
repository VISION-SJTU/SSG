how_many=30000
ref_data="coco2014"
ref_dir="coco2014"
ref_type="val2014"
eval_res=256
batch_size=512
clip_model="ViT-B/32"

caption_file='captions_coco2014.txt'
fake_dir='baseline_ssg'

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m calculate_metrics --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size --clip_model $clip_model --caption_file $caption_file
