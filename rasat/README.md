# The official implementation of the paper "RASAT: Integrating Relational Structures into Pretrained Seq2Seq Model for Text-to-SQL"(EMNLP 2022)

This is the official implementation of the following paper:

Jiexing Qi and Jingyao Tang and Ziwei He and Xiangpeng Wan and Yu Cheng and Chenghu Zhou and Xinbing Wang and Quanshi Zhang and Zhouhan Lin. RASAT: Integrating Relational Structures into Pretrained Seq2Seq Model for Text-to-SQL. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP).

If you use this code, please cite:

```
@article{Qi2022RASATIR,
  title={RASAT: Integrating Relational Structures into Pretrained Seq2Seq Model for Text-to-SQL},
  author={Jiexing Qi and Jingyao Tang and Ziwei He and Xiangpeng Wan and Yu Cheng and Chenghu Zhou and Xinbing Wang and Quanshi Zhang and Zhouhan Lin},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.06983}
}
```


# Quick start

## Code downloading
This repository uses git submodules. Clone it like this:

```
$ git clone https://github.com/JiexingQi/RASAT.git
$ cd RASAT
$ git submodule update --init --recursive
```
## Download the dataset
Before running the code, you should download dataset files.

First, you should create a dictionary like this:
```
mkdir -p dataset_files/ori_dataset
```

And then you need to download the dataset file to dataset_files/ and just keep it in zip format. The download links are here:
+ Spider, [link](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0)
+ SParC, [link](https://drive.google.com/uc?export=download&id=13Abvu5SUMSP3SJM-ZIj66mOkeyAquR73)
+ CoSQL, [link](https://drive.google.com/uc?export=download&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP)

Then unzip those dataset files into dataset_files/ori_dataset. Both files in zip format and unzip format is needed:

```
unzip dataset_files/spider.zip -d dataset_files/ori_dataset/
unzip dataset_files/cosql_dataset.zip -d dataset_files/ori_dataset/
unzip dataset_files/sparc.zip -d dataset_files/ori_dataset/
```

## The Coreference Resolution Files
We recommend you just use the generated coreference resolution files. It just needs you run

```
unzip preprocessed_dataset.zip -d ./dataset_files/
```

If you want to generate these coreference resolution files by yourself, you could create a new conda environment to install coreferee library since it may have a version conflict with other libraries. The install commands are as follows:

```bash
conda create -n coreferee python=3.9.7
conda activate coreferee
bash run_corefer_processing.sh
```

and you can just assign the dataset name and the corresponding split, such as 
```
python3 get_coref.py --input_path ./cosql_dataset/sql_state_tracking/cosql_dev.json --output_path ./dev_coref.json --dataset_name cosql --mode dev
```

## Environment setup

### Use docker
The best performance is achieved by exploiting PICARD[1], and if you want to reproduce it, we recommend you use Docker.

You can simply use 
```
make eval
```
to start a new docker container for an interaction terminal that supports PICARD. 

Since the docker environment doesn't have stanza, so you should run these commands before training or evaluting:
```
pip install stanza
python3 seq2seq/stanza_downloader.py
```

**Note:We only use PICARD for seperately evalutaion.**

### Do not use Docker
If Docker is not available to you, you could also run it in a python 3.9.7 environment 

```bash
conda create -n rasat python=3.9.7
conda activate rasat
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install -r requirements.txt
```

However, you could not use PICARD in that way.

**Please Note: the version of stanza must keep 1.3.0, other versions will lead to error. **




## Training

You can simply run these code like this:

- Single-GPU
```
CUDA_VISIBLE_DEVICES="0" python3 seq2seq/run_seq2seq.py configs/sparc/train_sparc_rasat_small.json
```

- Multi-GPU 
```
CUDA_VISIBLE_DEVICES="2,3" python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 seq2seq/run_seq2seq.py configs/sparc/train_sparc_rasat_small.json
```

and you should set --nproc_per_node=#gpus to make full use of all GPUs. A recommend total_batch_size = #gpus * gradient_accumulation_steps * per_device_train_batch_size is 2048.


## Evalutaion

You can simply run these codes:

```
CUDA_VISIBLE_DEVICES="2" python3 seq2seq/eval_run_seq2seq.py configs/cosql/eval_cosql_rasat_576.json
```

Notice：If you use Docker for evaluation, you may need to change the filemode for these dictionary before starting a new docker container:

```
chmod -R 777 seq2seq/
chmod -R 777 dataset_files/
```


# Result and checkpoint

The models shown below use database content, and the corresponding column like "edge_type", and "use_coref" are parameters set in config.json. All these model checkpoints are available in Huggingface. 

## CoSQL
| model                                              | edge_type | use_dependency | use_coref | QEM/IEM(Dev) | QEX/IEX(Dev) | QEM/IEM(Test) | QEX/IEX(Test) |
|---------------------------------------------------:|-----------|----------------|-----------|--------------|--------------|---------------|---------------|
| Jiexing/cosql_add_coref_t5_3b_order_0519_ckpt-576  | Default   | FALSE          | TRUE      | 56.1/25.9    | 63.2/34.1    | -             | -             |
| + PICARD                                           | Default   | FALSE          | TRUE      | 58.6/27.0    | 67.0/39.6    | 53.6/24.1     | 64.9/34.3     |
| Jiexing/cosql_add_coref_t5_3b_order_0519_ckpt-2624 | Default   | FALSE          | TRUE      | 56.4/25.6    | 63.1/34.8    | -             | -             |
| + PICARD                                           | Default   | FALSE          | TRUE      | 57.9/26.3    | 66.1/38.6    | **55.7/26.5**     | **66.3/37.4**     |




## SParC
| model                                              | edge_type | use_dependency | use_coref | QEM/IEM(Dev) | QEX/IEX(Dev) | QEM/IEM(Test) | QEX/IEX(Test) |
|---------------------------------------------------:|-----------|----------------|-----------|--------------|--------------|---------------|---------------|
| Jiexing/sparc_add_coref_t5_3b_order_0514_ckpt-4224 | Default   | FALSE          | TRUE      | 65.0/45.5    | 69.9/50.7    | -             | -             |
| + PICARD                                           | Default   | FALSE          | TRUE      | 67.5/46.9    | 73.2/53.8    | 67.7/44.9     | 74.0/52.6     |
| Jiexing/sparc_add_coref_t5_3b_order_0514_ckpt-5696 | Default   | FALSE          | TRUE      | 63.7/47.4    | 68.1/50.2    | -             | -             |
| + PICARD                                           | Default   | FALSE          | TRUE      | 67.1/49.3    | 72.5/53.6    | 67.3/45.2     | 73.6/52.6     |



## Spider

| model                              | edge_type | use_dependency | use_coref | EM(Dev) | EX(Dev) | EM(Test) | EX(Test) |
|-----------------------------------:|-----------|----------------|-----------|---------|---------|----------|----------|
| Jiexing/spider_relation_t5_3b-2624 | Default   | FALSE          | FALSE     | 72      | 76.6    | -        | -        |
|                           + PICARD | Default   | FALSE          | FALSE     | 74.7    | **80.5**    | 70.6     | **75.5**     |
| Jiexing/spider_relation_t5_3b-4160 | Default   | FALSE          | FALSE     | 72.6    | 76.6    | -        | -        |
|                           + PICARD | Default   | FALSE          | FALSE     | **75.3**    | 78.3    | **70.9**     | 74.5     |


# Acknowledgements
We would like to thank Tao Yu, Hongjin Su, and Yusen Zhang for running evaluations on our submitted models. We would also like to thank Lyuwen Wu for her comments on the Readme file of our code repository.
