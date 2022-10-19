CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task3/config.yaml \
    --input_dir ../../data/task3/data/probing/$2/template1 \
    --mask_position all \
    --output_dir task3/output \
    --recompute 


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task3/config.yaml \
    --input_dir ../../data/task3/data/probing/$2/template1 \
    --output_dir task3/output \
    --mask_position concept 




