import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from evaluation.compute_metrics import computeMetrics
from src.dataset import same_poi_multiview_dataset
from src.M3GAL.model import create_M3GAL
from src.utils import (
    check_make_path,
    ddp_setup,
    get_tensorboard_writer,
    helper_print,
    setup_seed,
    AverageMeter,
    ProgressMeter,
)

parser = argparse.ArgumentParser("")
# general deep learning parameters
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training"
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
parser.add_argument(
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument(
    "-b",
    "--batch_size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--gc-lr",
    "--gc-learning-rate",
    default=0.0001,
    type=float,
    metavar="GC_LR",
    help="initial gc learning rate",
    dest="gc_lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum of SGD solver",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)


def main_worker(gpu, ngpus_per_node, args, config):
    helper_print("S1, init environment.")
    ddp_setup(gpu, args)
    if args.seed is not None:
        setup_seed(args.seed)

    print(">>>>> args params before construct model in GPU:", args.gpu)
    print("args:", json.dumps(args.__dict__, indent=4))
    print("config:", json.dumps(config, indent=4))

    # create model
    helper_print("S2, create model.")
    model = create_M3GAL(args)
    if args.gpu is not None:
        model = model.cuda()
    if args.multiprocessing_distributed:
        model = DDP(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    gc_encoder_params = list(map(id, model.gc_encoder.parameters()))
    gc_encoder_shadow_params = list(map(id, model.gc_encoder_shadow.parameters()))
    base_params = filter(
        lambda p: id(p) not in gc_encoder_params
        and id(p) not in gc_encoder_shadow_params,
        model.parameters(),
    )

    optimizer = torch.optim.SGD(
        [
            {
                "name": "gc",
                "params": model.gc_encoder.parameters(),
                "lr": args.gc_lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            },
            {
                "name": "base",
                "params": base_params,
                "lr": args.lr,
                "momentum": args.momentum,
                "weigth_decay": args.weight_decay,
            },
        ],
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    helper_print("S3, dataloader")
    # Data loading code
    train_dataset = same_poi_multiview_dataset(config["same_poi_multiview_file"])
    val_dataset = torch.load(config["val_dataset_filepath"])
    test_dataset = torch.load(config["test_dataset_filepath"])
    print(f"Length f train dataset : {len(train_dataset)}")
    print(f"Length f val dataset : {len(val_dataset)}")
    print(f"Length f test dataset : {len(test_dataset)}")

    if args.multiprocessing_distributed:
        same_poi_multiview_sampler = torch.utils.data.distributed.DistributedSampler(
            same_poi_multiview_dataset
        )
    else:
        same_poi_multiview_sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(same_poi_multiview_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=same_poi_multiview_sampler,
        drop_last=True,
    )
    val_loader = val_dataset
    test_loader = test_dataset

    # tensorboard
    writer = get_tensorboard_writer(args, f"tensorboard/{args.output_name}/")

    helper_print("S5, traning")
    best_recall1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_momentum(model, args)

        if args.multiprocessing_distributed:
            same_poi_multiview_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for on epoch
        # dict{"losses":xx,"top1":xx,"top5":}
        train_metrics = train(
            train_dataloader, model, criterion, optimizer, epoch, args
        )

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.global_rank == 0
        ):
            # evaluate
            # dict{recall@1,recall@3,recall@5,mrr@1,mrr@3,mrr@5}
            val_addr_metrics, val_gc_metrics, val_metrics = evaluate(
                model, val_loader, config, args.output_name
            )
            test_addr_metrics, test_gc_metrics, test_metrics = evaluate(
                model, test_loader, config, args.output_name
            )
            # tensorboard
            if writer:
                for key in [
                    "addr_losses",
                    "addr_top1",
                    "addr_top5",
                    "gc_losses",
                    "gc_top1",
                    "gc_top5",
                    "losses",
                    "top1",
                    "top5",
                ]:
                    writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                for key in [
                    "recall@1",
                    "recall@3",
                    "recall@5",
                    "mrr@1",
                    "mrr@3",
                    "mrr@5",
                ]:
                    writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
                    writer.add_scalar(f"test/{key}", test_metrics[key], epoch)
                    writer.add_scalar(f"val/addr_{key}", val_addr_metrics[key], epoch)
                    writer.add_scalar(f"test/addr_{key}", test_addr_metrics[key], epoch)
                    writer.add_scalar(f"val/gc_{key}", val_gc_metrics[key], epoch)
                    writer.add_scalar(f"test/gc_{key}", test_gc_metrics[key], epoch)

            # save checkpoint
            if (epoch + 1) % args.save_every == 0:
                if args.output_name:
                    save_path = "checkpoints/{}/checkpoint_{:04d}.pth.tar".format(
                        args.output_name, epoch
                    )
                else:
                    save_path = "checkpoints/checkpoint_{:04d}.pth.tar".format(epoch)
                check_make_path(save_path)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.text_encoder,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=save_path,
                )
            # save best
            if val_metrics["recall@1"] > best_recall1:
                best_recall1 = val_metrics["recall@1"]
                if args.output_name:
                    save_path = "checkpoints/{}/checkpoint_{:04d}_best.pth.tar".format(
                        args.output_name, epoch
                    )
                else:
                    save_path = "checkpoints/checkpoint_{:04d}_best.pth.tar".format(
                        epoch
                    )
                check_make_path(save_path)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.text_encoder,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=save_path,
                )

    if writer:
        writer.close()


def adjust_momentum(model, args):
    model.gc_m = min(model.gc_m + args.m_inc, args.moco_gc_m)


@torch.no_grad()
def evaluate(model, data_loader, config, output_name):
    model.eval()

    gold_max = []
    total = 0
    querys = []
    docs = []
    lngs = []
    lats = []
    cand_size = []
    scores = []
    gc_scores = []

    for datas in data_loader:
        querys.append(datas["query"])
        # current_doc = ["" for _ in range(40)]
        for did in range(len(datas["docs"])):
            docs.append(datas["docs"][did])
            lngs.append((datas["lngs"][did] - config["lng_mean"]) / config["lng_std"])
            lats.append((datas["lats"][did] - config["lat_mean"]) / config["lat_std"])
        cand_size.append(len(datas["docs"]))
        gold_max.append(datas["gold_max"])
        total += 1

    # shape of query_embedding: 1 x 768
    # shape of docs_embedding: 40*768
    if model is torch.nn.parallel.distributed.DistributedDataParallel:
        model = model.module

    eval_batch_size = 100
    querys_embedding = []
    for i in tqdm(range(0, len(querys), eval_batch_size)):
        querys_embedding.append(
            model.compute_embedding(querys[i : i + eval_batch_size], type="q")
        )
    querys_embedding = torch.concat(querys_embedding, dim=0)
    docs_embedding = []
    for i in tqdm(range(0, len(docs), eval_batch_size)):
        docs_embedding.append(
            model.compute_embedding(docs[i : i + eval_batch_size], type="k")
        )
    docs_embedding = torch.concat(docs_embedding, dim=0)

    gcs_embedding = []
    for i in tqdm(range(0, len(docs), eval_batch_size)):
        gcs_embedding.append(
            model.compute_embedding(
                np.stack(
                    [lngs[i : i + eval_batch_size], lats[i : i + eval_batch_size]],
                    axis=1,
                ),
                type="gc_shadow",
            )
        )
    gcs_embedding = torch.concat(gcs_embedding, dim=0)
    start_idx = 0
    for i, query_embedding in enumerate(querys_embedding):
        score = torch.einsum(
            "nc,kc->nk",
            [
                torch.unsqueeze(query_embedding, 0),
                docs_embedding[start_idx : start_idx + cand_size[i]],
            ],
        ).squeeze()
        scores.append(score)
        gc_score = torch.einsum(
            "nc,kc->nk",
            [
                torch.unsqueeze(query_embedding, 0),
                gcs_embedding[start_idx : start_idx + cand_size[i]],
            ],
        ).squeeze()
        gc_scores.append(gc_score)
        start_idx += cand_size[i]

    rank = torch.argsort(torch.stack(scores), dim=1, descending=True).cpu().numpy() + 1
    gc_rank = (
        torch.argsort(torch.stack(gc_scores), dim=1, descending=True).cpu().numpy() + 1
    )
    total_rank = (
        torch.argsort(
            torch.stack(scores) + torch.stack(gc_scores), dim=1, descending=True
        )
        .cpu()
        .numpy()
        + 1
    )
    # print(rank.shape)
    if output_name:
        save_path = "output/{}/textgeoalign_rerank_detail.npy".format(output_name)
    else:
        save_path = "output/textgeoalign_rerank_detail.npy"
    check_make_path(save_path)

    np.save(
        save_path,
        {
            "rank": rank,
            "gold_max": gold_max,
            "gc_rank": gc_rank,
            "total_rank": total_rank,
        },
    )

    metrics = computeMetrics(gold_max=gold_max, model_ranks=rank)
    print(metrics)
    gc_metrics = computeMetrics(gold_max=gold_max, model_ranks=gc_rank)
    print(gc_metrics)
    total_metrics = computeMetrics(gold_max=gold_max, model_ranks=total_rank)
    print(total_metrics)
    return metrics, gc_metrics, total_metrics


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch, args):
    # time meter
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    # losses meter
    losses = AverageMeter("Losses", ":.4e")
    addr_losses = AverageMeter("Loss(Address)", ":.4e")
    gc_losses = AverageMeter("Loss(GC)", ":.4e")
    # top1 meter
    top1 = AverageMeter("Acc@1", ":6.2f")
    addr_top1 = AverageMeter("Acc(Address)@1", ":6.2f")
    gc_top1 = AverageMeter("Acc(GC)@1", ":6.2f")
    # top5 meter
    top5 = AverageMeter("Acc@5", ":6.2f")
    addr_top5 = AverageMeter("Acc(Address)@5", ":6.2f")
    gc_top5 = AverageMeter("Acc(GC)@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time,
            data_time,
            losses,
            addr_losses,
            gc_losses,
            top1,
            addr_top1,
            gc_top1,
            top5,
            addr_top5,
            gc_top5,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # print(optimizer)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (addr1, addr2, lng, lat, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        lng = (lng - config["lng_mean"]) / config["lng_std"]
        lat = (lat - config["lat_mean"]) / config["lat_std"]
        geolocation = torch.stack([lng.float(), lat.float()], dim=1)
        # check data consistency
        if i == 0:
            helper_print(addr1[0], addr2[0], lng[0], lat[0])

        # compute flops
        addr_output, gc_output, target = model(
            addr_q=addr1, addr_k=addr2, geolocation=geolocation
        )
        addr_loss = criterion(addr_output, target)
        gc_loss = criterion(gc_output, target)
        loss = criterion(addr_output + gc_output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        addr_acc1, addr_acc5 = accuracy(addr_output, target, topk=(1, 5))
        gc_acc1, gc_acc5 = accuracy(gc_output, target, topk=(1, 5))
        acc1, acc5 = accuracy(addr_output + gc_output, target, topk=(1, 5))

        current_bs = len(addr1)
        addr_losses.update(addr_loss.item(), current_bs)
        gc_losses.update(gc_loss.item(), current_bs)
        losses.update(loss.item(), current_bs)
        addr_top1.update(addr_acc1[0], current_bs)
        gc_top1.update(gc_acc1[0], current_bs)
        top1.update(acc1[0], current_bs)
        addr_top5.update(addr_acc5[0], current_bs)
        gc_top5.update(gc_acc5[0], current_bs)
        top5.update(acc5[0], current_bs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.loss == "addr_loss":
            loss.backward()
        elif args.loss == "gc_loss":
            gc_loss.backward()
        elif args.loss == "total_loss":
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return {
        "addr_losses": addr_losses.avg,
        "addr_top1": addr_top1.avg,
        "addr_top5": addr_top5.avg,
        "gc_losses": gc_losses.avg,
        "gc_top1": gc_top1.avg,
        "gc_top5": gc_top5.avg,
        "losses": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
    }


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    gc_lr = args.gc_lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
        gc_lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        if param_group["name"] == "gc":
            param_group["lr"] = gc_lr
        else:
            param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args, config):
    if args.multiprocessing_distributed:
        # 集群内的机器数
        args.nodes = int(os.environ["WORLD_SIZE"])
        # 当前是第几台机器
        args.node_rank = int(os.environ["RANK"])
        # 每台机器的GPU数量
        args.ngpus_per_node = torch.cuda.device_count()
        # 主机地址
        args.master_address = (
            "tcp://" + os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
        )
        # 整个集群有多少卡
        args.global_world_size = args.ngpus_per_node * args.nodes
        args.batch_size = int(args.batch_size / args.global_world_size)

    if args.multiprocessing_distributed:
        mp.spawn(
            main_worker,
            nprocs=args.ngpus_per_node,
            args=(args.ngpus_per_node, args, config),
        )
    else:
        main_worker(args.gpu, 1, args, config)


if __name__ == "__main__":
    parser.add_argument("--text_encoder", default="bert-base-chinese")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_samples", default=None, type=int)
    parser.add_argument("--create_dataset", action="store_true")
    parser.add_argument("--bert_from_local", action="store_false")

    parser.add_argument("--save_every", default=1000, type=int)
    parser.add_argument("--close_tensorboard", action="store_true", help="")
    parser.add_argument(
        "--output_name", default="default", type=str, help="part of save path."
    )
    parser.add_argument(
        "--data_suffix", default=None, type=str, help="part of save path."
    )
    parser.add_argument(
        "--test_data_suffix", default=None, type=str, help="test on which scale"
    )
    parser.add_argument("--test", action="store_true", help="")

    # moco specific configs:
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
        "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
    )
    parser.add_argument(
        "--loss",
        default="loss",
        type=str,
        help="choose from [addr_loss, gc_loss, total_loss]",
    )

    parser.add_argument(
        "--m_inc",
        default=0,
        type=float,
        help="gc_m is initialized as 0, then it is increased by m_inc each epoch util it reachs args.gc_m",
    )
    parser.add_argument(
        "--normalize",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="whether use norm for embeddings in moco",
    )
    args = parser.parse_args()

    config = yaml.load(open("./src/config.yaml", "r"), Loader=yaml.Loader)

    if args.data_suffix:
        config[
            "same_poi_multiview_file"
        ] = f"data/same_poi_views_{args.data_suffix}.csv"
    if args.test_data_suffix:
        config[
            "val_dataset_filepath"
        ] = f"data/val_dataset_use_gis_{args.test_data_suffix}.pth"
        config[
            "test_dataset_filepath"
        ] = f"data/test_dataset_use_gis_{args.test_data_suffix}.pth"
    if args.test:
        config["val_file"] = "data/MGeo/val_dataset_1000.pth"
        config["test_file"] = "data/MGeo/test_dataset_1000.pth"
        config["same_addr_file"] = "data/same_address_pairs_10k.npy"
        config["diff_addr_file"] = "data/different_address_pairs_10k.npy"

    main(args, config)
