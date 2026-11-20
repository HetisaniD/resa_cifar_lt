# k-nearest-neighbour classification on frozen features.
#
# A test image is assigned the class that wins a vote among its k nearest
# training images ("the bank") in feature space. No training is involved, so
# k-NN measures the representation directly: it cannot learn a class prior
# the way a linear layer trained with cross-entropy can.
#
# Three things are configurable:
#
#   k              number of neighbours (the study uses 5, 10 and 20).
#   metric         'euclidean' on the raw features (the protocol of the ReSA
#                  CIFAR code) or 'cosine' on L2-normalised features.
#   class_weight   None: every neighbour has one vote (plain majority vote).
#                  An array w[c]: a neighbour of class c votes with weight
#                  w[c]. With w[c] = 1 / n_c (n_c = bank images of class c)
#                  this is the RE-BALANCED vote: k-NN estimates the class
#                  density as (neighbours of class c) / n_c, so dividing by
#                  n_c removes the class prior of a long-tailed bank -- the
#                  k-NN counterpart of class-balanced sampling for the
#                  linear probe.
#
# Ties are broken in favour of the class whose member is nearest to the
# query. (The quick monitor in eval/evaluate.py resolves ties by lowest class
# index, which on a long-tailed dataset favours the most frequent classes.)

import numpy as np
import torch


@torch.no_grad()
def knn_classify(bank_x, bank_y, query_x, k_values, num_classes, metric="euclidean",
                 class_weight=None, device="cpu", chunk=1000):
    """Classify `query_x` with one k-NN vote per value in `k_values`.

    bank_x [N, D], bank_y [N]: features and labels of the training images.
    query_x [M, D]: features of the test images.
    Returns {k: (predictions int64 [M], vote shares float32 [M, C])}; the vote
    shares sum to 1 per row and serve as class scores (e.g. for AUC).
    """
    k_values = sorted(set(int(k) for k in k_values))
    k_max = min(max(k_values), len(bank_x))

    bank = torch.as_tensor(bank_x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(bank_y, dtype=torch.long, device=device)
    query = torch.as_tensor(query_x, dtype=torch.float32)
    if metric == "cosine":
        bank = torch.nn.functional.normalize(bank, dim=1)
    elif metric != "euclidean":
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
    weight = None
    if class_weight is not None:
        weight = torch.as_tensor(np.asarray(class_weight), dtype=torch.float64, device=device)

    preds = {k: [] for k in k_values}
    shares = {k: [] for k in k_values}
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk].to(device)
        if metric == "cosine":
            q = torch.nn.functional.normalize(q, dim=1)
            idx = (q @ bank.T).topk(k_max, dim=1, largest=True, sorted=True).indices
        else:
            idx = torch.cdist(q, bank).topk(k_max, dim=1, largest=False, sorted=True).indices
        neigh = labels[idx]  # [m, k_max] neighbour labels, nearest first

        for k in k_values:
            kk = min(k, k_max)
            lab = neigh[:, :kk]
            w = torch.ones(lab.shape, dtype=torch.float64, device=device) if weight is None else weight[lab]
            votes = torch.zeros(len(lab), num_classes, dtype=torch.float64, device=device)
            votes.scatter_add_(1, lab, w)

            # tie-break: rank of each class's nearest member (0 = nearest)
            rank = torch.arange(kk, dtype=torch.float64, device=device).expand_as(lab)
            best = torch.full((len(lab), num_classes), float(kk), dtype=torch.float64, device=device)
            best.scatter_reduce_(1, lab, rank, reduce="amin")
            eps = votes.max(dim=1, keepdim=True).values * 1e-6 / kk
            preds[k].append((votes + eps * (kk - best)).argmax(dim=1).cpu())
            shares[k].append((votes / votes.sum(dim=1, keepdim=True)).float().cpu())

    return {k: (torch.cat(preds[k]).numpy(), torch.cat(shares[k]).numpy()) for k in k_values}
