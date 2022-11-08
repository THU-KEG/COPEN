# COPEN
COPEN is a COnceptual knowledge Porobing bENchmark introduced in the paper ``COPEN: Probing Conceptual Knowledge in Pre-trained Language Models''.
COPEN consists of three tasks: (1) Conceptual Similarity Judgment (CSJ). Given a query entity and several candidate entities, the CSJ task requires to 
select the most conceptually similar candidate entity to the query entity. (2) Conceptual Property Judgment (CPJ). Given a statement describing a property of 
a concept, PLMs need to judge whether the statement is true. (3) Conceptualization in Contexts (CiC). Given a sentence, an entity mentioned in the sentence, and several concept chains of the entity, PLMs need to select the most appropriate concept according to the context for the entity. 

# Quick Start
The code repository is based on `Pytorch` and `Transformers`. Please use the following command to install all 
the necessary dependcies.
`pip install -r requirements.txt`

## Download Datasets
The COPEN benchmark is placed on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/f0b33fb429fa4575aa7f/?dl=1), please use the following command to download the datasets and place them
in the propor path.
```shell
cd data/
wget --content-disposition https://cloud.tsinghua.edu.cn/f/f0b33fb429fa4575aa7f/?dl=1
unzip copen_data.zip 
mv copen_data/task1/ task1/data
mv copen_data/task2/ task2/data
mv copen_data/task3/ task3/data 
```

## Pre-processing Datasets
### Probing
```shell
cd task1
python probing_data_processor.py
cd ../
cd task2
python data_processor_for_ppl.py
cd ../
cd task3
python probing_data_processor.py
cd ../
```

### Fine-tuning
```shell
python processor_utils.py task1 mc 
python processor_utils.py task2 sc
python processor_utils.py task3 mc 
```

## Run 
### Probing
```shell
cd code/probing
bash task1/run.sh 0 bert bert-base-uncased
bash task2/run.sh 0 bert bert-base-uncased
bash task3/run.sh 0 bert bert-base-uncased
```

### Fine-Tuning
```shell
cd code/finetuning
cd task1/ 
bash ../run.sh 0 bert bert-base-uncased task1 mc 42
cd task2/ 
bash ../run.sh 0 bert bert-base-uncased task2 sc 42
cd task3/ 
bash ../run.sh 0 bert bert-base-uncased task3 mc 42
```

