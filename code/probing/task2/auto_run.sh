bash task2/run.sh $1 bert prajjwal1/bert-small 
bash task2/run.sh $1 bert prajjwal1/bert-medium 
bash task2/run.sh $1 bert bert-base-uncased 
bash task2/run.sh $1 bert bert-large-uncased 

bash task2/run.sh $1 roberta roberta-base 

bash task2/run.sh $1 gpt2 gpt2  
bash task2/run.sh $1 gpt2 gpt2-medium 
bash task2/run.sh $1 gpt2 gpt2-large
bash task2/run.sh $1 gpt2 gpt2-xl 

bash task2/run.sh $1 gpt_neo EleutherAI/gpt-neo-125M

bash task2/run.sh $1 bart facebook/bart-base  

bash task2/run.sh $1 t5 t5-small
bash task2/run.sh $1 t5 t5-base 
bash task2/run.sh $1 t5 t5-large

# bash task2/run.sh $1 t5 t5-3b


# bash task2/run.sh $1 t5 t5-11b




