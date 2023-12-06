import pandas as pd
import argparse
import yaml
import torch
from src.M3GAL.model import create_M3GAL
from src.M3GAL.moco_train import evaluate
import numpy as np
import json


def load_checkpoint(model, args):
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    if args.gpu is None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(args.checkpoint, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            args.checkpoint, checkpoint["epoch"]
        )
    )


def compute_query_embeddings(model, test_filepath):
    querys = []
    for i, line in enumerate(open(test_filepath)):
        querys.append(json.loads(line)["query"])

    query_embeddings = []
    for i in range(0, len(querys), 100):
        query_embeddings.append(
            model.compute_embedding(querys[i : min(i + 100, len(querys))], type="q")
        )

    query_embeddings = torch.concat(query_embeddings, axis=0)
    print(query_embeddings.shape)
    return query_embeddings


def prepare_targets(test_filepath, poi_database_filepath):
    targets = []
    poi_database_df = pd.read_csv(poi_database_filepath)
    poi_database_list = [
        (x[0], np.around(x[1], 6), np.around(x[2], 6)) for x in poi_database_df.values
    ]
    for i, line in enumerate(open(test_filepath)):
        sample = json.loads(line)
        positive_poi = sample["positive_passages"][0]
        gis = positive_poi["gis"][-1]
        gis = np.fromstring(gis, dtype=float, sep=",")
        target_poi = (positive_poi["text"], np.around(gis[0], 6), np.around(gis[1], 6))
        assert target_poi in poi_database_list, target_poi
        targets.append(poi_database_list.index(target_poi) + 1)
    np.save("data/test_targets.npy", targets)
    print(
        "data/test_targets.npy saved, you can use it by add --target_filepath=data/test_targets.npy next time"
    )
    return targets


def compute_poi_embeddings(model, poi_database_csv, config):
    df = pd.read_csv(poi_database_csv)
    addresses = df["address"].to_list()
    lng = (df["lng"].to_numpy() - config["lng_mean"]) / config["lng_std"]
    lat = (df["lat"].to_numpy() - config["lat_mean"]) / config["lat_std"]
    gl = np.stack((lng, lat), axis=1)

    address_embeddings = []
    gl_embeddings = []
    for i in range(0, len(addresses), 100):
        address_embeddings.append(
            model.compute_embedding(
                addresses[i : min(i + 100, len(addresses))], type="k"
            )
        )
        gl_embeddings.append(
            model.compute_embedding(
                gl[i : min(i + 100, len(addresses)), :], type="gc_shadow"
            )
        )

    address_embeddings = torch.concat(address_embeddings, axis=0)
    gl_embeddings = torch.concat(gl_embeddings, axis=0)
    poi_embeddings = torch.concat([address_embeddings, gl_embeddings], axis=1)
    np.save("data/mgeo_poi_database.embeddings", poi_embeddings.cpu().numpy())
    print(
        "data/mgeo_poi_database.embeddings saved, you can use it by adding poi_embeddings_filepath=data/mgeo_poi_database.embeddings next time"
    )
    return poi_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate on query-POI retrieval problem")
    parser.add_argument("--text_encoder", default="bert-base-chinese")
    parser.add_argument("--bert_from_local", action="store_false")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="best model checkpoint"
    )
    parser.add_argument(
        "--test_dataset_filepath",
        default=None,
        type=str,
        help="test data file",
    )
    parser.add_argument(
        "--moco-dim", default=768, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--moco-k",
        default=4096,
        type=int,
        help="queue size; number of negative keys (default: 4096)",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-gc-m",
        default=0.999,
        type=float,
        help="moco momentum of update gc encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-geohash-m",
        default=0.999,
        type=float,
        help="moco momentum of update geohash key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
    )
    parser.add_argument(
        "--normalize",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="whether use norm for embeddings in moco",
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--output_name", default="test", type=str, help="output directory"
    )

    args = parser.parse_args()

    config = yaml.load(open("src/config.yaml", "r"), Loader=yaml.FullLoader)

    # Create model
    model = create_M3GAL(args)

    # Load checkpoint
    load_checkpoint(model, args)

    model.eval()
    if args.test_dataset_filepath is not None:
        config["test_dataset_filepath"] = args.test_dataset_filepath
    test_dataset = torch.load(args.test_dataset_filepath)
    test_loader = test_dataset
    test_addr_metrics, test_gc_metrics, test_metrics = evaluate(
        model, test_loader, config, args.output_name
    )
