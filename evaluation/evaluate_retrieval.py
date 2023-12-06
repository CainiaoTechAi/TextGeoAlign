import pandas as pd
import argparse
import yaml
import torch
from src.M3GAL.model import create_M3GAL
import numpy as np
import json
from evaluation.compute_metrics import computeMetrics
import shutil
import os


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
    np.save("data/mgeo_query.embeddings", query_embeddings.cpu().numpy())
    print(
        "data/mgeo_query.embeddings saved, you can use it by adding --query_embeddings_filepath=data/mgeo_query.embeddings.npy next time"
    )
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


def compute_ranks(query_embeddings, poi_embeddings, args):
    if os.path.exists("output/addr_ranks"):
        shutil.rmtree("output/addr_ranks")
    os.mkdir("output/addr_ranks")
    if os.path.exists("output/gc_ranks"):
        shutil.rmtree("output/gc_ranks")
    os.mkdir("output/gc_ranks")
    if os.path.exists("output/ranks"):
        shutil.rmtree("output/ranks")
    os.mkdir("output/ranks")
    batch_size = 200
    n_parts = (query_embeddings.shape[0] - 1) // batch_size + 1
    for i in range(n_parts):
        start = batch_size * i
        end = min(batch_size * (i + 1), query_embeddings.shape[0])
        addr_scores = torch.matmul(
            query_embeddings[start:end], poi_embeddings[:, :768].T
        )
        gc_scores = torch.matmul(query_embeddings[start:end], poi_embeddings[:, 768:].T)
        addr_ranks = torch.argsort(addr_scores, axis=1, descending=True) + 1
        gc_ranks = torch.argsort(gc_scores, axis=1, descending=True) + 1
        ranks = torch.argsort(addr_scores + gc_scores, axis=1, descending=True) + 1
        if i == 0 or i == n_parts - 1:
            print(
                f"Shape of addr_score {addr_scores.shape}",
            )
            print(f"Shape of gc_scores {gc_scores.shape}")
            print(f"Shape of ranks {ranks.shape}")

        torch.save(addr_ranks[:, :100].cpu(), f"output/addr_ranks/part{i+1}.pth")
        torch.save(gc_ranks[:, :100].cpu(), f"output/gc_ranks/part{i+1}.pth")
        torch.save(ranks[:, :100].cpu(), f"output/ranks/part{i+1}.pth")
        print(f"part{i+1} saved")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate on query-POI retrieval problem")
    parser.add_argument("--text_encoder", default="bert-base-chinese")
    parser.add_argument("--bert_from_local", action="store_false")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--compute_ranks", action="store_true")
    parser.add_argument(
        "--poi_database_filepath", default="data/mgeo_poi_database.csv", type=str
    )
    parser.add_argument(
        "--poi_embeddings_filepath",
        type=str,
        default=None,
        help="poi embedding file, if None, recompute",
    )
    parser.add_argument(
        "--query_embeddings_filepath",
        type=str,
        default=None,
        help="query embedding file, if None, recompute",
    )
    parser.add_argument(
        "--target_filepath", type=str, default=None, help="test target file"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="best model checkpoint"
    )
    parser.add_argument(
        "--test_dataset_filepath",
        default="data/test_dataset.jsonl",
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

    args = parser.parse_args()

    config = yaml.load(open("src/config.yaml", "r"), Loader=yaml.FullLoader)

    # if the embeddings are prepared, we don't need to load model
    if args.load_model:
        # Create model
        model = create_M3GAL(args)

        # Load checkpoint
        load_checkpoint(model, args)

        model.eval()

    # compute poi embeddings
    if args.compute_ranks:
        if args.poi_embeddings_filepath is None:
            poi_embeddings = compute_poi_embeddings(
                model, args.poi_database_filepath, config
            )
            print("POI embedding are computed")

        else:
            poi_embeddings = torch.tensor(np.load(args.poi_embeddings_filepath))
            if args.gpu is not None:
                poi_embeddings = poi_embeddings.to("cuda:0")
        print(f"Shape of poi database embeddings {poi_embeddings.shape}")

    # prepare test targets
    if args.target_filepath is None:
        targets = prepare_targets(
            args.test_dataset_filepath, args.poi_database_filepath
        )
    else:
        targets = np.load(args.target_filepath)
    print("Targets are computed")
    print(f"length of targets : {len(targets)}")
    # print(targets)

    # compute query embeddings
    if args.compute_ranks:
        if args.query_embeddings_filepath is None:
            query_embeddings = compute_query_embeddings(
                model, args.test_dataset_filepath
            )
        else:
            query_embeddings = torch.tensor(np.load(args.query_embeddings_filepath))
            if args.gpu is not None:
                query_embeddings = query_embeddings.to("cuda:0")

        print(
            f"Shape of query embeddings {query_embeddings.shape}",
        )

    # compute scores and ranks
    # poi_len = poi_embeddings.shape[0]
    # addr_scores = []
    # gc_scores = []
    # # it would OOM when the poi database is large, thus we compute the score batch by batch
    # for i in tqdm(range(0, poi_len, 10)):
    #     addr_scores.append(torch.matmul(query_embeddings, torch.tensor(poi_embeddings[i:min(i+10, poi_len), :768]).to("cuda:0").T).cpu())
    #     gc_scores.append(torch.matmul(query_embeddings, torch.tensor(poi_embeddings[i:min(i+10, poi_len), 768:]).to("cuda:0").T).cpu())
    #     torch.cuda.empty_cache()
    # addr_scores = torch.concat(addr_scores, axis=1)
    # gc_scores = torch.concat(gc_scores, axis=1)

    if args.compute_ranks:
        compute_ranks(query_embeddings, poi_embeddings, args)

    k = 1
    ranks = []
    addr_ranks = []
    gc_ranks = []
    while os.path.exists(f"output/ranks/part{k}.pth"):
        print(f"output/ranks/part{k}.pth loaded")
        ranks.append(torch.load(f"output/ranks/part{k}.pth"))
        gc_ranks.append(torch.load(f"output/gc_ranks/part{k}.pth"))
        addr_ranks.append(torch.load(f"output/addr_ranks/part{k}.pth"))
        k += 1
        print(gc_ranks[-1].shape)
    print(f"{k-1} rank files are loaded")
    ranks = torch.concat(ranks, axis=0)
    addr_ranks = torch.concat(addr_ranks, axis=0)
    gc_ranks = torch.concat(gc_ranks, axis=0)
    print(f"shape of ransk {ranks.shape}")
    print(f"shape of addr_ranks {addr_ranks.shape}")
    print(f"shape of gc_ranks {gc_ranks.shape}")

    # get evaluation result
    addr_metrics = computeMetrics(targets, addr_ranks)
    print(addr_metrics)
    gc_metrics = computeMetrics(targets, gc_ranks)
    print(gc_metrics)
    metrics = computeMetrics(targets, ranks)
    print(metrics)
