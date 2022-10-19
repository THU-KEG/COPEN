GPU_ID=$1
MODEL_TYPE=$2
MODEL_NAME_OR_PATH=$3
TASK_NAME=$4
TASK_TYPE=$5
SEED=$6

if [ "$MODEL_TYPE" == "t5" ]; then 
    SCRIPT=main_for_seq2seq.py
    CONF_PATH="t5.yaml"
else 
    SCRIPT=main.py 
    CONF_PATH="config.yaml"
fi 

echo $GPU_ID
echo $MODEL_TYPE
echo $MODEL_NAME_OR_PATH
echo $TASK_NAME
echo $TASK_TYPE
echo $SCRIPT
echo $SEED

# CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
#     --conf_path configs/$CONF_PATH \
#     --seed $SEED \
#     --model_type $MODEL_TYPE \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --train_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/train.jsonl \
#     --validation_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/dev.jsonl \
#     --test_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/test.jsonl \
#     --output_dir checkpoint/$SEED/full/$MODEL_NAME_OR_PATH-iid \
#     --early_stopping_patience 5 

CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
    --conf_path configs/$CONF_PATH \
    --seed $SEED \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/train.jsonl \
    --validation_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/dev.jsonl \
    --test_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/test.jsonl \
    --output_dir checkpoint/$SEED/full/$MODEL_NAME_OR_PATH-ood \
    --early_stopping_patience 5 

# CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
#     --conf_path configs/$CONF_PATH \
#     --seed $SEED \
#     --model_type $MODEL_TYPE \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --train_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/train.jsonl \
#     --validation_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/dev.jsonl \
#     --test_file ../../../data/$TASK_NAME/data/iid/$TASK_TYPE/$MODEL_TYPE/test.jsonl \
#     --output_dir checkpoint/$SEED/linear/$MODEL_NAME_OR_PATH-iid \
#     --early_stopping_patience 7 \
#     --learning_rate 1e-3 \
#     --num_train_epochs 20 \
#     --freeze_backbone_parameters 

# CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
#     --conf_path configs/$CONF_PATH \
#     --seed $SEED \
#     --model_type $MODEL_TYPE \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --train_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/train.jsonl \
#     --validation_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/dev.jsonl \
#     --test_file ../../../data/$TASK_NAME/data/ood/$TASK_TYPE/$MODEL_TYPE/test.jsonl \
#     --output_dir checkpoint/$SEED/linear/$MODEL_NAME_OR_PATH-ood \
#     --early_stopping_patience 7 \
#     --learning_rate 1e-3 \
#     --num_train_epochs 20 \
#     --freeze_backbone_parameters 