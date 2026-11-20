# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

# ReSA: the self-supervised objective that is pretrained in this study.
# (Weng et al., "Clustering Properties of Self-Supervised Learning", ICML 2025.
# This file is the authors' implementation; only comments and the device
# handling were added.)
#
# One training step, for a batch of B images with two augmented views each:
#   1. The online encoder gives features h (512-d for ResNet-18) and the
#      projector gives embeddings emb, for both views.
#   2. A momentum (EMA) copy of encoder and projector does the same without
#      gradients: h_m, emb_m.
#   3. The B x B cosine-similarity matrix between the ENCODER outputs
#      h and h_m is turned into a soft assignment matrix by Sinkhorn-Knopp.
#      Row i says which images of the batch image i is considered to be
#      "the same thing as" -- the target is produced by the model's own
#      encoder, hence "positive feedback" / self-assignment.
#   4. The loss is the cross-entropy between that assignment and the softmax
#      of the EMBEDDING similarities (temperature --temperature) across the
#      two views.
#
# Why this matters under class imbalance: Sinkhorn-Knopp forces the
# assignment matrix to have equal row and column sums, i.e. every image of
# the batch receives the same total weight. With a long-tailed batch the
# head classes fill most of the batch, so a head image has many true
# neighbours to share weight with while a tail image has few or none.
#
# The momentum encoder is the network that is saved for evaluation
# (state-dict prefix "momentum_encoder.").

import torch
import copy
from itertools import chain
from method.base import BaseMethod

class ReSA(BaseMethod):
    # ReSA using the momentum network, better performance

    def __init__(self, args):
        super().__init__(args)

        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        
        for param in chain(self.momentum_encoder.parameters(), 
                           self.momentum_projector.parameters()):
            param.requires_grad = False

        self.temp = args.temperature

    def forward(self, samples):

        # move every view to the model's device (GPU in practice; CPU for smoke tests)
        device = next(self.encoder.parameters()).device
        samples = [x.to(device, non_blocking=True) for x in samples]

        h, emb = self.ForwardWrapper(samples, self.encoder, self.projector)
        
        with torch.no_grad():
            # EMA update, then targets from the momentum network (no gradient)
            self.update_momentum_params(self.momentum)
            h_m, emb_m = self.ForwardWrapper(samples[:2], self.momentum_encoder, self.momentum_projector)
            # B x B assignment from encoder-output similarities (step 3 above)
            assign = self.sinkhorn_knopp(h @ h_m.T)
           
        total_loss = 0
        n_loss_terms = 0

        # swapped prediction: view q of the online network predicts the
        # assignment using view v != q of the momentum network (step 4)
        for q in range(len(emb)):
            for v in range(len(emb_m)):
                if v == q:
                    continue
                emb_sim = emb[q] @ emb_m[v].T / self.temp
                total_loss += self.cross_entropy(emb_sim, assign)
                n_loss_terms += 1
        return total_loss / n_loss_terms



