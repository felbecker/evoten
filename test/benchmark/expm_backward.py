#!/usr/bin/env python3
"""Benchmark the GTR matrix-exponential gradient: Daleckii-Krein vs. autograd.

Three ways to get d/dparams of P(t) = exp(Q t) for reversible Q:

  dk     evoten.expm_gtr_torch.expm_gtr: one eigh per rate matrix, custom backward with the
         Daleckii-Krein divided differences (what Agrippa trains with)
  eigh   the same eigendecomposition route, but autograd through torch.linalg.eigh
  pade   autograd through torch.linalg.matrix_exp(Q t), one Pade exponential per branch

at Agrippa's codon-stack size: M = 16 rate matrices of 64 x 64, times = the 126 branch
lengths of the 64-leaf tree. Q is built from kernels exactly as in training
(softplus exchangeabilities, softmax equilibrium, normalised to one substitution), and the
loss is <P, G> for a fixed random G, so the gradients w.r.t. the exchangeability kernel,
the equilibrium kernel and the branch lengths are those of a real step.

Stability: share of non-finite gradient entries and relative error (Frobenius norm) against
autograd through matrix_exp in float64, for four kinds of rate matrices:
  random     random kernels, distinct eigenvalues
  baseline   12 random + 4 exact Jukes-Cantor (Agrippa's 4 neutral codon components start as JC)
  nearJC-5   all 16 near Jukes-Cantor, kernel jitter 1e-5
  nearJC-9   all 16 near Jukes-Cantor, kernel jitter 1e-9 (the end state of Agrippa sweep 5)
Speed: median wall time of forward and forward+backward in float32 on the GPU, peak memory.

  python test/benchmark/expm_backward.py --tree preswall64.nwk [--out results.txt]
"""
import argparse
import json
import statistics
import time

import torch

from evoten.backend_pytorch import BackendTorch
from evoten.expm_gtr_torch import _finish, expm_gtr

B = BackendTorch()


def branch_lengths(newick_path):
    from Bio import Phylo
    tree = Phylo.read(newick_path, "newick")
    bl = [c.branch_length for c in tree.find_clades() if c is not tree.root and c.branch_length]
    return torch.tensor(bl, dtype=torch.float64)


def expm_eigh_autograd(Q, t, pi):
    """expm_gtr's eigen route with autograd through torch.linalg.eigh."""
    sqrt_pi = torch.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi
    S = Q * inv_sqrt_pi.unsqueeze(-2) * sqrt_pi.unsqueeze(-1)
    S = 0.5 * (S + S.transpose(-1, -2))
    lam, U = torch.linalg.eigh(S)                                # (M, d), (M, d, d)
    g = torch.exp(lam * t.unsqueeze(-1))                         # (T, M, d)
    M = (U * g.unsqueeze(-2)) @ U.transpose(-1, -2)              # (T, M, d, d)
    P = inv_sqrt_pi.unsqueeze(-1) * M * sqrt_pi.unsqueeze(-2)
    return _finish(P, t.broadcast_to(P.shape[:-2]))


def expm_pade(Q, t, pi):
    P = torch.linalg.matrix_exp(Q[None] * t[..., None, None])
    return _finish(P, t.broadcast_to(P.shape[:-2]))


METHODS = {"dk": expm_gtr, "eigh": expm_eigh_autograd, "pade": expm_pade}


def kernels(kind, M, d, gen):
    k0 = float(B.inverse_softplus(torch.tensor(1.0, dtype=torch.float64)))
    rnd = torch.randn(M, d, d, dtype=torch.float64, generator=gen)
    pik = 0.5 * torch.randn(M, d, dtype=torch.float64, generator=gen)
    if kind == "random":
        return rnd, pik
    if kind == "baseline":
        K, P = rnd.clone(), pik.clone()
        K[12:] = k0
        P[12:] = 0.0
        return K, P
    jitter = {"nearJC-5": 1e-5, "nearJC-9": 1e-9}[kind]
    return k0 + jitter * rnd, torch.zeros(M, d, dtype=torch.float64)


def grads(method, K, pik, t, G):
    K = K.clone().requires_grad_(True)
    pik = pik.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    pi = B.make_equilibrium(pik)
    Q = B.make_rate_matrix(B.make_symmetric_pos_semidefinite(K), pi)
    P = METHODS[method](Q, t[:, None], pi)
    (P * G).sum().backward()
    return {"K": K.grad, "pi": pik.grad, "t": t.grad}


def stability(K, pik, t, G, device, dtype):
    cast = dict(device=device, dtype=dtype)
    ref = grads("pade", K, pik, t, G)                            # float64 on the CPU
    rows = {}
    for m in METHODS:
        try:
            g = grads(m, K.to(**cast), pik.to(**cast), t.to(**cast), G.to(**cast))
        except (torch.linalg.LinAlgError, RuntimeError) as err:      # e.g. eigh not converging
            rows[m] = {name: {"nonfinite": float("nan"), "rel_err": float("nan"), "error": str(err)[:120]}
                       for name in ("K", "pi", "t")}
            continue
        row = {}
        for name, v in g.items():
            v = v.double().cpu()
            nonfinite = float((~torch.isfinite(v)).double().mean())
            rel = float((v - ref[name]).norm() / ref[name].norm()) if nonfinite == 0 else float("nan")
            row[name] = {"nonfinite": nonfinite, "rel_err": rel}
        rows[m] = row
    return rows


def timing(method, K, pik, t, G, reps, warmup):
    def step(backward):
        Kr = K.clone().requires_grad_(backward)
        pr = pik.clone().requires_grad_(backward)
        tr = t.clone().requires_grad_(backward)
        pi = B.make_equilibrium(pr)
        Q = B.make_rate_matrix(B.make_symmetric_pos_semidefinite(Kr), pi)
        P = METHODS[method](Q, tr[:, None], pi)
        if backward:
            (P * G).sum().backward()
        else:
            P.sum().item()

    out = {}
    for backward in (False, True):
        for _ in range(warmup):
            step(backward)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            step(backward)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        out["fwd+bwd" if backward else "fwd"] = {
            "median_ms": 1000 * statistics.median(ts),
            "peak_MB": torch.cuda.max_memory_allocated() / 2**20}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="Newick file; its branch lengths are the times")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None, help="also write the report (and a .json) here")
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(0)
    t = branch_lengths(args.tree)
    G = torch.randn(len(t), args.M, args.d, args.d, dtype=torch.float64, generator=gen)
    lines = [f"M={args.M} rate matrices of {args.d}x{args.d}, {len(t)} branch lengths "
             f"(min {t.min():.4f}, median {t.median():.4f}, max {t.max():.4f}) from {args.tree}",
             f"torch {torch.__version__}, "
             f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'no GPU'}"]
    result = {"setup": lines[:], "stability": {}, "timing": {}}

    kinds = ["random", "baseline", "nearJC-5", "nearJC-9"]
    places = [("cpu", torch.float64)]
    if torch.cuda.is_available():
        places.append(("cuda", torch.float32))
    lines.append("\nSTABILITY: share of non-finite gradient entries / relative error vs. "
                 "matrix_exp autograd in float64")
    for kind in kinds:
        K, pik = kernels(kind, args.M, args.d, gen)
        for device, dtype in places:
            key = f"{kind} {device} {str(dtype).split('.')[-1]}"
            rows = stability(K, pik, t, G, device, dtype)
            result["stability"][key] = rows
            lines.append(f"  {key}")
            for m, row in rows.items():
                cells = "  ".join(f"{n}: {r['nonfinite']:6.1%} nonfinite, rel.err {r['rel_err']:.1e}"
                                  for n, r in row.items())
                lines.append(f"    {m:5s} {cells}")

    if torch.cuda.is_available():
        lines.append(f"\nTIMING: float32 on the GPU, median of {args.reps}")
        for kind in ("random", "baseline"):
            K, pik = kernels(kind, args.M, args.d, gen)
            dev = dict(device="cuda", dtype=torch.float32)
            Kc, pc, tc, Gc = K.to(**dev), pik.to(**dev), t.to(**dev), G.to(**dev)
            for m in METHODS:
                r = timing(m, Kc, pc, tc, Gc, args.reps, args.warmup)
                result["timing"][f"{kind} {m}"] = r
                lines.append(f"  {kind:8s} {m:5s} fwd {r['fwd']['median_ms']:7.2f} ms  "
                             f"fwd+bwd {r['fwd+bwd']['median_ms']:7.2f} ms  "
                             f"peak {r['fwd+bwd']['peak_MB']:7.0f} MB")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
