GPU_ID=$1
TASK_NAME=$2
TASK_TYPE=$3
SEED=$4

# bash ../run.sh $GPU_ID bert prajjwal1/bert-small $TASK_NAME $TASK_TYPE $SEED
# bash ../run.sh $GPU_ID bert prajjwal1/bert-medium $TASK_NAME $TASK_TYPE $SEED
bash ../run.sh $GPU_ID bert bert-base-uncased $TASK_NAME $TASK_TYPE $SEED
# bash ../run.sh $GPU_ID bert bert-large-uncased $TASK_NAME $TASK_TYPE $SEED

# bash ../run.sh $GPU_ID roberta roberta-base $TASK_NAME $TASK_TYPE $SEED

# bash ../run.sh $GPU_ID gpt2 gpt2  $TASK_NAME $TASK_TYPE $SEED
# bash ../run.sh $GPU_ID gpt2 gpt2-medium $TASK_NAME $TASK_TYPE $SEED
# bash ../run.sh $GPU_ID gpt2 gpt2-large $TASK_NAME $TASK_TYPE $SEED
# bash ../run.sh $GPU_ID gpt2 gpt2-xl $TASK_NAME $TASK_TYPE $SEED

# bash ../run.sh $GPU_ID gpt_neo EleutherAI/gpt-neo-125M  $TASK_NAME $TASK_TYPE $SEED

# bash ../run.sh $GPU_ID bart facebook/bart-base  $TASK_NAME $TASK_TYPE $SEED

# bash ../run.sh $GPU_ID t5 t5-small $TASK_NAME qa $SEED
# bash ../run.sh $GPU_ID t5 t5-base $TASK_NAME qa $SEED
# bash ../run.sh $GPU_ID t5 t5-large $TASK_NAME qa $SEED





