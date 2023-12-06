import json
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def create_loader(
    datasets, samplers, batch_size, num_workers, is_trains, collate_fns, shuffle=None
):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            if shuffle is None:
                shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


def create_dataset(dataset, config, args):
    same_addr_pair_dataset = rerank_same_address_dataset(
        config["same_addr_file"], n_samples=args.n_samples
    )
    diff_addr_pair_dataset = rerank_diff_address_dataset(
        config["diff_addr_file"], n_samples=args.n_samples
    )
    return same_addr_pair_dataset, diff_addr_pair_dataset


def create_multiview_dataset(dataset, config, args):
    return same_poi_multiview_dataset(config["same_poi_multiview_file"])


class osm_biencoder_dataset(Dataset):
    def __init__(
        self, ann_file, max_words=64, use_query_gis=False, use_gis=False, n_samples=None
    ):
        self.max_words = max_words
        self.use_query_gis = use_query_gis
        self.use_gis = use_gis

        df = pd.read_csv(ann_file)
        self.querys = list(df["address1"])
        self.gold_max = [1 for _ in range(len(df))]
        formal_address = df["address2"]
        self.docs = []
        for i in range(len(df)):
            self.docs.append([formal_address[i]] + list(formal_address.sample(39)))

    def __len__(self):
        return len(self.querys)

    def __getitem__(self, index: Any) -> Any:
        datas = {
            "query": self.querys[index],
            "docs": self.docs[index],
            "gold_max": self.gold_max[index],
        }
        return datas


class rerank_train_dataset(Dataset):
    def __init__(
        self, ann_file, max_words=64, use_query_gis=False, use_gis=False, n_samples=None
    ):
        self.max_words = max_words
        self.use_query_gis = use_query_gis
        self.use_gis = use_gis
        self.ann = []

        for line in open(ann_file):
            data = json.loads(line)
            self.ann.append(data)
        if n_samples is not None:
            self.ann = self.ann[:n_samples]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        if self.use_gis:
            datas = {
                "query": info["query"][:64],
                "docs": [],
                "lngs": [],
                "lats": [],
                "gold_max": len(info["positive_passages"]),
            }
        else:
            datas = {
                "query": info["query"][:64],
                "docs": [],
                "gold_max": len(info["positive_passages"]),
            }

        for item in info["positive_passages"] + info["negative_passages"]:
            text = item["text"]
            datas["docs"].append(text[: self.max_words])
            if self.use_gis:
                gis = item["gis"][-1]
                gis = np.fromstring(gis, dtype=float, sep=",")
                datas["lngs"].append(gis[0])
                datas["lats"].append(gis[1])
        return datas


class rerank_diff_address_dataset(Dataset):
    def __init__(self, dataset_file, max_words=64, n_samples=None):
        self.max_words = max_words
        self.data = np.load(dataset_file)

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        address1, address2, degree, distance = self.data[index]
        # if len(address1) > self.max_words:
        #     address1 = address1[-self.max_words :]
        # if len(address2) > self.max_words:
        #     address2 = address2[-self.max_words :]
        address1 = address1[:64]
        address2 = address2[:64]
        return address1, address2, degree, distance


class two_task_dataset(Dataset):
    def __init__(
        self, same_address_file, diff_address_file, max_words=64, n_samples=None
    ) -> None:
        super().__init__()
        self.max_words = max_words
        self.same_address_data = np.load(same_address_file)
        self.diff_address_data = np.load(diff_address_file)

        if n_samples is not None:
            self.same_address_data = self.same_address_data[:n_samples]
            self.diff_address_data = self.diff_address_data[:n_samples]

    def __len__(self):
        return len(self.same_address_data) + len(self.diff_address_data)

    def __getitem__(self, index: Any) -> Any:
        if index >= len(self.same_address_data):
            index -= len(self.same_address_data)
            address1, address2, degree, distance = self.diff_address_data[index]
            return address1[:64], address2[:64], float(1.0)
        else:
            address1, address2 = self.same_address_data[index]
            return address1[:64], address2[:64], float(0.0)


class poi_db_dataset(Dataset):
    def __init__(self, dataset_file, max_words=64, n_samples=None):
        self.max_words = max_words
        self.data = pd.read_csv(dataset_file)
        if n_samples is not None:
            self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list:
        (addr, lng, lat) = self.data.iloc[index]
        return addr, lng, lat


class same_poi_multiview_dataset(Dataset):
    def __init__(self, dataset_file, max_words=64, n_samples=None):
        self.max_words = max_words
        self.data = pd.read_csv(dataset_file)

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> list:
        (
            address1,
            address2,
            lng,
            lat,
            geohash1,
            geohash2,
            geohash3,
            geohash4,
            geohash5,
            geohash6,
            geohash7,
        ) = list(self.data.iloc[index])
        geohash = np.array(
            [geohash1, geohash2, geohash3, geohash4, geohash5, geohash6, geohash7]
        )
        return (address1, address2, lng, lat, geohash)


class rerank_same_address_dataset(Dataset):
    def __init__(self, dataset_file, max_words=64, n_samples=None):
        self.max_words = max_words
        self.data = np.load(dataset_file)

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        address1, address2 = self.data[index]
        # if len(address1) > self.max_words:
        #     address1 = address1[-self.max_words :]
        # if len(address2) > self.max_words:
        #     address2 = address2[-self.max_words :]
        address1 = address1[:64]
        address2 = address2[:64]
        return address1, address2
