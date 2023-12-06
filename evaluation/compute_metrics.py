import numpy as np


def computeRecallTopK(gold_max, model_ranks, k=None):
    assert k > 0 or k is None, "k should be a positive integer"
    hitted = 0
    for i, answer in enumerate(gold_max):
        if k is None:
            k = len(model_ranks[i])
        if answer in model_ranks[i, :k]:
            hitted += 1
    return hitted / len(gold_max)


def computeMRRTopK(gold_max, model_ranks, k=None):
    assert k > 0 or k is None, "k should be a positive integer"
    mrr = 0
    for i, answer in enumerate(gold_max):
        if k is None:
            k = len(model_ranks[i])
        for pos, x in enumerate(model_ranks[i, :k]):
            if x == answer:
                mrr += 1 / (pos + 1)
    return mrr / len(gold_max)


def computeMetrics(gold_max, model_ranks):
    metrics = {}

    metrics["recall@1"] = computeRecallTopK(gold_max, model_ranks, 1)
    metrics["recall@3"] = computeRecallTopK(gold_max, model_ranks, 3)
    metrics["recall@5"] = computeRecallTopK(gold_max, model_ranks, 5)
    metrics["mrr@1"] = computeMRRTopK(gold_max, model_ranks, 1)
    metrics["mrr@3"] = computeMRRTopK(gold_max, model_ranks, 3)
    metrics["mrr@5"] = computeMRRTopK(gold_max, model_ranks, 5)

    return metrics


if __name__ == "__main__":
    rerank_detail = np.load("output/rerank_detail.npy", allow_pickle=True)[()]
    gold_max = rerank_detail["gold_max"]
    model_ranks = rerank_detail["rank"]
    metrics = computeMetrics(gold_max, model_ranks)
    print(metrics)
    # for markdown
    print("| recall@1 | recall@3 | recall@5 | mrr@1 | mrr@3 | mrr@5 |")
    print("| ---- | ---- | ---- | ---- | ---- | ----| ")
    print(
        f"| {metrics['recall@1']} \
        | {metrics['recall@3']} | \
        {metrics['recall@5']} | \
        {metrics['mrr@1']} | \
        {metrics['mrr@3']} | \
        {metrics['mrr@5']} |"
    )
