[TOC]

## M3GAL

Momentum Multi-modal Geographical Alignment for Query-POI matching

### Download

1. **OSM-HZ Multiview Dataset**: A dataset offering multiview descriptions of locations within Hangzhou, China, generated from OpenStreetMap data (Comming soon).
2. **GeoTES Dataset**: An open-source Query-POI Matching dataset, accessible at https://github.com/PhantomGrapes/MGeo
3. **Pretrained Model**: The pretrained model for M3GAL (Comming soon).
4. **Finetuned Model**: The model finetuned based on M3GAL (Comming soon).

```plaintext
root/
├── data/
│ ├── same_poi_views_100k.csv							# pretrain
│ ├── train_dataset_use_gis.pth						# finetune
│ ├── test_dataset_use_gis_1000.pth				# pretrain,finetune
│ ├── val_dataset_use_gis_1000.pth				# pretrain,finetune
│ ├── mgeo_poi_database_train_use_negative.csv	# finetune
│ ├── same_poi_views_mgeo.csv							# finetune
│ ├── mgeo_poi_database_use_negative.csv	# retrieval
│ ├── test_dataset_use_gis_20000.pth			# ranking
│ └── test_dataset.jsonl									# retrieval,ranking
└── checkpoints/
  └── M3GAL_pretrained/
		└── checkpoint_0027_best.pth.tar			# checkpoint
```



### Pretrain M3GAL on OSM dataset

```
python -m src.M3GAL.moco_train --epochs=100 --lr=0.001 --batch_size=1024 --gc-lr=0.00001 --moco-m=0.999 --moco-gc-m=0.999 --moco-k=16384 --print-freq=10 --data_suffix=100k --loss=total_loss --output_name=M3GAL_pretrained --test_data_suffix=1000 --m_inc=0.01 --schedule 40 80 --gpu=0 --bert_from_local --text_encoder='/root/data/huggingface/bert-base-chinese' --seed=2023
```

### Finetuned on GeoTES Dataset

#### Bi-Encoder Finetuned for Ranking

```
python -m src.M3GAL.biencoder_finetune --epochs=30 --lr=0.0001 --batch_size=64 --gc-lr=0.0001 --moco-m=0.999 --moco-gc-m=0.999 --moco-k=16384 --print-freq=10 --output_name=M3GAL_finetune_biencoder --schedule 10 20 --gpu=0 --resume=checkpoints/M3GAL_pretrained/checkpoint_0027_best.pth.tar --bert_from_local --text_encoder='/root/data/huggingface/bert-base-chinese' --seed=202312
```

### Evaluation

#### Evaluation - Retrieval

```
python -m evaluation.evaluate_retrieval --moco-k=16384 --checkpoint=checkpoints/M3GAL_pretrained/checkpoint_0027_best.pth.tar --gpu=0 --poi_database_filepath=data/mgeo_poi_database_use_negative.csv --compute_ranks --load_model --bert_from_local --text_encoder='/root/data/huggingface/bert-base-chinese'
```

#### Evaluation - Ranking

```
python -m evaluation.evaluate_rerank --moco-k=16384 --checkpoint=checkpoints/M3GAL_pretrained/checkpoint_0027_best.pth.tar --test_dataset_filepath=data/test_dataset_use_gis_20000.pth --gpu=0 --bert_from_local --text_encoder='/root/data/huggingface/bert-base-chinese' 
```

### Citation

