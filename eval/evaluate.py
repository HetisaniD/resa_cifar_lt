# Evaluation run every --eval_every epochs DURING pretraining (main.py).
#
# It is a quick monitor of representation quality, following the W-MSE / INTL
# protocol used by the ReSA paper for CIFAR: features of the frozen momentum
# encoder, a 5-nearest-neighbour vote and a linear classifier trained with
# Adam for 500 epochs. For a long-tailed --dataset the classifier is trained
# on the long-tailed split, so this monitor inherits the head-class bias
# analysed in the study. The numbers reported in the dissertation come from
# the dedicated scripts instead: eval_linear_cifar.py, eval_knn.py and
# eval_transfer.py.

import torch
import torch.nn as nn
import torch.optim as optim
from .dataloader import get_data

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_acc(model, ds, args):
    model.eval()
    model = unwrap_model(model)

    if hasattr(model, 'momentum_encoder'):
        feature = model.momentum_encoder
    else:
        feature = model.encoder
    out_size = model.out_size

    device = next(feature.parameters()).device
    x_train, y_train = get_data(feature, ds.clf, out_size, device)
    x_test, y_test = get_data(feature, ds.test, out_size, device)

    acc_knn = eval_knn(x_train, y_train, x_test, y_test)
    acc_linear = eval_sgd(x_train, y_train, x_test, y_test)

    del x_train, y_train, x_test, y_test
    model.train()
    return acc_knn, acc_linear

def eval_knn(x_train, y_train, x_test, y_test, k=5, chunk_size=1000):

  """ k-nearest neighbors classifier accuracy.

  Computed in chunks over x_test rather than a single torch.cdist(x_test,
  x_train) call. A single call materializes the full (len(x_test),
  len(x_train)) distance matrix at once -- fine for CIFAR-10's 10,000-row
  test set, but for CINIC-10's 90,000-row test set against even a modest
  ~45,000-row long-tailed train pool that's 90000*45000*4 bytes ~= 16GB,
  which is what was OOMing here. Chunking never allocates more than
  (chunk_size, len(x_train)) at once and gives bit-identical predictions,
  since each test row's k-nearest-neighbor vote only depends on its own
  distances to x_train, never on other test rows.
  """
  pred = torch.empty_like(y_test)
  for start in range(0, len(x_test), chunk_size):
      end = start + chunk_size
      d = torch.cdist(x_test[start:end], x_train)
      topk = torch.topk(d, k=k, dim=1, largest=False)
      labels = y_train[topk.indices]
      for i in range(len(labels)):
          x = labels[i].unique(return_counts=True)
          pred[start + i] = x[0][x[1].argmax()]
      del d, topk, labels
  acc = (pred == y_test).float().mean().cpu().item()
  del pred
  return acc

def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500):
    """ linear classifier accuracy (sgd) """
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.to(x_train.device)
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        # perm.split(1000) instead of the previous
        # torch.randperm(len(x_train)).view(-1, 1000): view() requires
        # len(x_train) to be an exact multiple of 1000, which a long-tailed
        # pool essentially never is (e.g. 44667 -> "shape '[-1, 1000]' is
        # invalid for input of size 44667"). split(1000) shuffles the same
        # way but allows a final, smaller batch, so every sample is still
        # used exactly once per epoch.
        perm = torch.randperm(len(x_train))
        for idx in perm.split(1000):
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {
        t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item()
        for t in topk
    }
    del clf
    return acc
