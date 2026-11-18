"""Figure 2 panels (a) and (b): does the victim's pruner remove the repair and keep the injection?

(a) fraction of REPAIR-support columns the victim's pruner removes   (paper: 94-100%)
(b) fraction of INJECTION-support columns that survive               (paper: within 2pp of the
    geometric ceilings, 87.5% at 30% sparsity and 62.5% at 2:4)

The adversary fixes supports from Wanda scores on the ORIGINAL W using its own corpus (WikiText-2).
The victim scores the RELEASED W* = W + Delta on a corpus of its own. Since the Wanda score is
|W|*||x||, the victim's score on W* is obtained from its score on W by rescaling |W*|/|W| -- no need
to recompute activations per edit.

No LLM judging: this is pure weight-space measurement.

Usage: support_overlap.py <model_short> <edit_dir> <attacker_metrics_dir> <victim_metrics_dir> \
                          <sparsity|2of4> [gamma_scale]
"""
import glob
import os
import re
import sys

import torch

from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.model import load_model

MODEL, EDIT, ATT, VIC, SPARSE = sys.argv[1:6]
GSCALE = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
RHO_INJ, RHO_REP = 0.8, 0.12


def prune_mask(metric, sparse):
    """True where the victim's pruner ZEROES the weight."""
    if sparse == "2of4":
        do, di = metric.shape
        g = metric[:, : di - di % 4].reshape(do, -1, 4)
        idx = g.argsort(dim=-1)[..., :2]                  # two smallest of every four
        m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True).reshape(do, -1)
        out = torch.zeros_like(metric, dtype=torch.bool)
        out[:, : m.shape[1]] = m
        return out
    tau = torch.quantile(metric.float(), float(sparse), dim=1, keepdim=True)
    return metric <= tau


def main():
    model, _ = load_model(MODEL)
    layers = []
    for f in sorted(glob.glob(os.path.join(EDIT, "L*_down_proj.pt"))):
        layers.append((int(re.search(r"L(\d+)_", os.path.basename(f)).group(1)), f))

    rep_removed, inj_survived, n = 0.0, 0.0, 0
    print(f"{'layer':>6s} {'(a) repair removed':>19s} {'(b) injection survived':>23s}")
    for L, ef in layers:
        name = f"model.layers.{L}.mlp.down_proj.weight.pt"
        att = torch.load(os.path.join(ATT, name), map_location="cpu").float()
        vic = torch.load(os.path.join(VIC, name), map_location="cpu").float()
        delta = torch.load(ef, map_location="cpu").float() * GSCALE

        # weights of this layer, to rescale the victim metric onto W*
        W = dict(model.named_parameters())[f"model.layers.{L}.mlp.down_proj.weight"]
        W = W.detach().float().cpu()
        vic_star = vic * (W + delta).abs() / (W.abs() + 1e-12)

        # adversary's supports, ranked on the ORIGINAL W with the adversary's corpus
        order = att.argsort(dim=1, descending=True)
        k = att.shape[1]
        n_inj, n_rep = int(RHO_INJ * k), int(RHO_REP * k)
        inj = torch.zeros_like(att, dtype=torch.bool).scatter_(1, order[:, :n_inj], True)
        rep = torch.zeros_like(att, dtype=torch.bool).scatter_(1, order[:, k - n_rep:], True)

        pruned = prune_mask(vic_star, SPARSE)
        a = (pruned & rep).sum().item() / max(rep.sum().item(), 1)
        b = (~pruned & inj).sum().item() / max(inj.sum().item(), 1)
        rep_removed += a; inj_survived += b; n += 1
        print(f"{L:6d} {100*a:18.1f}% {100*b:22.1f}%")

    print(f"{'MEAN':>6s} {100*rep_removed/n:18.1f}% {100*inj_survived/n:22.1f}%")
    ceil = 87.5 if SPARSE != "2of4" else 62.5
    print(f"\n  geometric ceiling for (b) at {SPARSE}: {ceil}%")


if __name__ == "__main__":
    main()
