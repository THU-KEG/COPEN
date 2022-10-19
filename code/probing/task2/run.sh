CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task2/config.yaml \
    --input_dir ../../data/task2/data/probing/$2/template1 \
    --mask_position all \
    --output_dir task2/output \
    --recompute 


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task2/config.yaml \
    --input_dir ../../data/task2/data/probing/$2/template1 \
    --mask_position answer \
    --output_dir task2/output 


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --model_type $2 \
    --model_name_or_path $3 \
    --conf_path task2/config.yaml \
    --input_dir ../../data/task2/data/probing/$2/template1 \
    --mask_position concept \
    --output_dir task2/output 





