"""Muon: SGD-momentum whose update matrix is orthogonalized before it lands.

Vendored from https://github.com/KellerJordan/Muon (the optimizer behind
the modded-nanogpt speedrun), with the distributed code paths removed --
this is a single-GPU script, and the sharding machinery in the original
exists only to split the Newton-Schulz work across ranks.

WHY THIS EXISTS
---------------
Adam scales every weight independently: each entry of the update is its
gradient divided by that entry's own running magnitude. For a weight
MATRIX that is a strange thing to do, because it ignores the matrix
structure entirely -- a gradient whose energy is concentrated in one or
two singular directions produces an update that is equally lopsided, so
most of the step goes into stretching a handful of directions rather
than into learning.

Muon takes the momentum-smoothed gradient and replaces it with the
nearest orthogonal matrix, which flattens the spectrum: every singular
direction gets a step of roughly the same size. The result is steepest
descent under the spectral norm rather than under a per-entry norm.

Measured on this model (8 layers, 512-wide, batch 64 x 1024, constant
LR, no warmup, held-out loss over 8 fixed batches from the tail of
shard 0):

    optimizer                       loss @ 700 steps    ms/step
    AdamW 3e-4 (the old default)          5.7615          523.8
    AdamW 6e-4                            5.8522          519.4
    AdamW 3e-4, betas (0.9, 0.95)         5.7158          521.2
    Muon 0.01                             4.7025          550.2
    Muon 0.02                             4.7046          563.3
    Muon 0.05                             6.5546          549.5

Muon reaches AdamW's 700-step loss at step 300: 2.33x fewer steps for
+5% wallclock, so ~2.2x faster to a given loss. Two controls say the win
is really orthogonalization and not something incidental -- 6e-4 is
WORSE than 3e-4, so the old baseline was not simply undertrained, and
switching Adam's beta2 to 0.95 on its own buys only 0.046 of the 1.06.

Note the shape of the LR curve: 0.01 and 0.02 are tied to within noise,
but 0.05 diverges -- its loss bottoms out around step 350 and then
climbs. There is a broad flat optimum with a cliff on the far side of
it, and with no warmup and no decay in this script's loop, 0.01 is the
side of the flat region to be standing on.

WHAT IT MUST NOT TOUCH
----------------------
Orthogonalization only makes sense for a matrix that acts as a linear
map between two spaces of features. That rules out:

  * the token and position embedding tables, which are lookups -- each
    row is an independent vector, and any given step touches only the
    rows whose tokens appeared in the batch. Orthogonalizing across
    50304 rows would couple them all together.
  * the output head (here the same tensor as the token embedding,
    because of weight tying, so this is the same exclusion),
  * everything 1D: biases and LayerNorm gains.

Those keep a standard Adam, which is what SingleDeviceMuonWithAuxAdam
below runs internally so the training loop still has one optimizer to
step and one state dict to checkpoint.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Replace G with (approximately) the nearest orthogonal matrix.

    The exact answer is U @ V.T where U, S, V = G.svd(), but an SVD per
    weight matrix per step would cost far more than the training step
    it is trying to improve. This is a quintic Newton-Schulz iteration
    instead: five iterations of a fixed polynomial, i.e. a handful of
    small matmuls, which is what makes the whole thing affordable.

    The coefficients are deliberately non-convergent. They are chosen to
    maximize the slope at zero, which is what lets five steps be enough
    to lift the small singular values; the price is that the iteration
    settles into the range ~(0.7, 1.3) rather than converging to exactly
    1. Empirically that spread does not hurt -- the point is to flatten
    the spectrum, not to normalize it perfectly.

    Runs in bfloat16. This is an approximation being computed to a few
    percent, so the extra mantissa bits of fp32 would buy nothing and
    cost bandwidth.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # The iteration works with X @ X.T, so transpose to keep that
    # product small: for a tall matrix it is cheaper to orthogonalize
    # the transpose and flip the result back.
    if G.size(-2) > G.size(-1):
        X = X.mT
    # The iteration is only stable for a spectral norm <= 1, and the
    # Frobenius norm is an upper bound on it that costs one reduction.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """One Muon step's worth of update, before the learning rate."""
    momentum.lerp_(grad, 1 - beta)
    # Nesterov: orthogonalize a gradient that has already been nudged
    # toward the momentum direction, rather than the momentum itself.
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters, unused here
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # An orthogonal matrix has all its singular values equal to 1, so
    # the update's scale no longer depends on the gradient's magnitude
    # -- but it does still depend on the matrix's shape. This factor
    # makes a non-square matrix take the same size step per output
    # feature as a square one would, so one learning rate can serve
    # every matrix in the model despite c_fc being 2048x512 and c_attn
    # being 1536x512.
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Plain Adam, for the parameters Muon must not touch."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)   # bias correction: both
    buf2c = buf2 / (1 - betas[1] ** step)   # buffers start at zero
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the hidden weight matrices, Adam for everything else.

    Takes param groups with a `use_muon` flag saying which is which; see
    the module docstring for what belongs on each side. Bundling both
    into one optimizer keeps the training loop and the checkpoint simple
    -- one .step(), one .state_dict().
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {
                    "params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
