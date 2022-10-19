CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task1/config.yaml \
    --input_dir ../../data/task1/data/probing/$2/template1 \
    --mask_position all \
    --output_dir task1/output \
    --recompute 


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task1/config.yaml \
    --input_dir ../../data/task1/data/probing/$2/template1 \
    --output_dir task1/output \
    --mask_position e1 


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task1/config.yaml \
    --input_dir ../../data/task1/data/probing/$2/template1 \
    --output_dir task1/output \
    --mask_position e2 




