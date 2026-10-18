#!/usr/bin/env python3
"""CPU-only reproduction tests for the papers behind the current comfytrain experiments (SN56 image tournament).

Every test reproduces a concrete number, identity or qualitative claim of one paper and prints PASS/FAIL together with
the paper reference (equation / section / table as numbered in the arXiv version named below). Tests marked [pemetaan]
check how our trainer (g56_comfy_train.py) maps onto the paper, not a claim of the paper itself.

Papers (full notes: riset/papers/*.md):
  EDM2    Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024, arXiv:2312.02696
          (equation numbers identical in v1 and v2; figure numbers quoted from the v1 HTML)
  MSTOCK  Jang, Yun, Han, "Model Stock: All we need is just a few fine-tuned models", ECCV 2024, arXiv:2403.19522v2
  LORAP   Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024, arXiv:2402.12354v2
  WISE    Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022, arXiv:2109.01903v3
  SOUPS   Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models ...", ICML 2022, arXiv:2203.05482v3
  DIFFFIT Xie et al., "DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient
          Fine-Tuning", ICCV 2023, arXiv:2304.06648v6 (appendix numbered as in the arXiv HTML: sections 6-10)
  BONN    Bonnaire, Urfin, Biroli, Mezard, "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical
          Regularization in Training", NeurIPS 2025, arXiv:2505.17638v2

Requirements: python3 + numpy + torch (no GPU, no scipy, no ComfyUI). Deterministic (fixed seeds). ~17 s with 4 threads, ~28 s with 1 (PT_THREADS).
Usage:
  python3 paper_tests.py              # all tests
  python3 paper_tests.py -k edm2      # tests whose name contains "edm2"
  python3 paper_tests.py --list
Inside the trainer image (the file is baked at /workspace/comfytrain/tools after a rebuild):
  docker run --rm --entrypoint python g56:comfy /workspace/comfytrain/tools/paper_tests.py
or without a rebuild (mount the host copy):
  docker run --rm -v /root/g56/comfytrain/tools:/t:ro --entrypoint python g56:comfy /t/paper_tests.py
Exit code: 0 if every selected test passed, 1 otherwise.
"""
import argparse
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

torch.set_num_threads(max(1, min(int(os.environ.get("PT_THREADS", "4")), os.cpu_count() or 1)))

REGISTRY = []


def register(name, ref):
    def deco(fn):
        REGISTRY.append((name, ref, fn))
        return fn
    return deco


class Check:
    def __init__(self):
        self.failed, self.info = [], []

    def ok(self, cond, msg):
        if not bool(cond):
            self.failed.append(msg)

    def note(self, msg):
        self.info.append(msg)


# ======================================================================================================================
# EDM2: power-function EMA and post-hoc EMA (paper §3, App. C; official code NVlabs/edm2 training/phema.py)
# ======================================================================================================================
def edm2_exp_to_std(gamma):
    """App. C.4 Eq. (123): sigma_rel = sqrt((g+1) / ((g+2)^2 (g+3)))."""
    g = np.float64(gamma)
    return float(np.sqrt((g + 1) / (g + 2) ** 2 / (g + 3)))


def edm2_std_to_exp(sigma_rel):
    """App. C.4 Eq. (126) + Algorithm 2 (v1 numbering): largest real root of g^3 + 7g^2 + (16 - s^-2) g + (12 - s^-2)."""
    t = np.float64(sigma_rel) ** -2
    return float(np.roots([1.0, 7.0, 16.0 - t, 12.0 - t]).real.max())


def edm2_pdp(t_a, g_a, t_b, g_b):
    """App. C.5 Eq. (151) (numerically robust form), symmetric in a<->b exactly as the official power_function_correlation."""
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, g_b, -g_a)
    t_max = np.maximum(t_a, t_b)
    return (g_a + 1) * (g_b + 1) * t_ratio ** t_exp / ((g_a + g_b + 1) * t_max)


def edm2_solve(t_i, g_i, t_r, g_r):
    """App. C.5 Eq. (141)-(143) / Algorithm 3: x = A^-1 b with a_ij = <p_i, p_j>, b_i = <p_i, p_r>.
    Returns (X, A, B). Unlike the official solve_posthoc_coefficients, X is NOT renormalised to column sums of 1."""
    rv = lambda x: np.float64(x).reshape(-1, 1)  # noqa: E731
    cv = lambda x: np.float64(x).reshape(1, -1)  # noqa: E731
    A = edm2_pdp(rv(t_i), rv(g_i), cv(t_i), cv(g_i))
    B = edm2_pdp(rv(t_i), rv(g_i), cv(t_r), cv(g_r))
    return np.linalg.solve(A, B), A, B


def edm2_power_weights(gamma, T):
    """Discrete weights of Eq. (1) for a piecewise-constant theta (App. C.3 convention theta(t)=theta(t_next) on
    (t_cur, t_next]): w_t = (t^(g+1) - (t-1)^(g+1)) / T^(g+1), t = 1..T. theta(0) gets weight 0."""
    t = np.arange(1, T + 1, dtype=np.float64)
    return (t ** (gamma + 1) - (t - 1) ** (gamma + 1)) / np.float64(T) ** (gamma + 1)


def traditional_ema_profile(beta, T):
    """App. C.2 Eq. (110)-(111) in steps: p(t) = (1-beta) beta^(T-t) for t>=1, the remaining mass beta^T on theta(0)
    (EMA initialised to the initial weights, exactly what _run_one does with ema = copy of the initial params)."""
    t = np.arange(0, T + 1, dtype=np.float64)
    p = (1 - beta) * beta ** (T - t)
    p[0] = beta ** T
    return p


def profile_sigma_rel(p):
    """App. C.1 Eq. (109): sigma_rel = sqrt(Var[t]) / T for t ~ p."""
    T = len(p) - 1
    t = np.arange(0, T + 1, dtype=np.float64)
    mu = float((p * t).sum())
    return math.sqrt(float((p * (t - mu) ** 2).sum())) / T


@register("edm2_sigma_rel_gamma", "EDM2 §3.1-3.2 (gamma=16.97/6.94 <-> sigma_rel 0.05/0.10), App. C.4 Eq. (122)-(126), "
                                  "Alg. 2, Fig. 23 (v1) legend pairs")
def t_edm2_sigma_rel_gamma(ck):
    pairs = [(0.05, 16.97), (0.10, 6.94), (0.15, 3.56), (0.20, 1.83), (0.25, 0.72)]
    for s, g_paper in pairs:
        g = edm2_std_to_exp(s)
        ck.ok(round(g, 2) == g_paper, f"sigma_rel {s}: gamma {g:.4f} does not round to the paper's {g_paper}")
        ck.ok(abs(edm2_exp_to_std(g) - s) < 1e-12, f"round trip failed for sigma_rel {s}")
        ck.ok(abs(edm2_exp_to_std(g_paper) - s) < 5e-4, f"Eq. (123) at the paper's gamma {g_paper} gives "
                                                          f"{edm2_exp_to_std(g_paper):.5f} != {s}")
        ck.note(f"sigma_rel {s:.2f} -> gamma {g:8.4f} (paper {g_paper}); mu_rel=(g+1)/(g+2)={(g + 1) / (g + 2):.4f}")
    # gamma > 0 <=> sigma_rel < 12^-0.5 (App. C.4 text after Eq. (126))
    ck.ok(abs(edm2_exp_to_std(0.0) - 12 ** -0.5) < 1e-15, "sigma_rel(gamma=0) != 12^-0.5")
    ck.ok(edm2_std_to_exp(0.30) < 0, "sigma_rel 0.30 > 12^-0.5 should need gamma < 0")
    ck.note(f"upper bound sigma_rel < 12^-0.5 = {12 ** -0.5:.4f} (gamma > 0)")


@register("edm2_power_ema_update", "EDM2 §3.1 Eq. (1)-(2), App. C.3 Eq. (112)-(119) and C.4 Eq. (121), (127)")
def t_edm2_power_ema_update(ck):
    rng = np.random.default_rng(0)
    T, D = 500, 8
    theta = rng.standard_normal((T + 1, D)).cumsum(0)
    for s in (0.05, 0.10, 0.20):
        g = edm2_std_to_exp(s)
        ema = rng.standard_normal(D) * 1e3          # arbitrary start: beta(1)=0 must erase it
        for t in range(1, T + 1):
            beta = (1 - 1 / t) ** (g + 1)          # Eq. (2) / Eq. (127) with Delta t = 1
            ema = beta * ema + (1 - beta) * theta[t]
        w = edm2_power_weights(g, T)
        closed = (w[:, None] * theta[1:]).sum(0)
        err = np.abs(ema - closed).max() / np.abs(closed).max()
        ck.ok(err < 1e-12, f"sigma_rel {s}: incremental update != Eq. (1) closed form (rel err {err:.2e})")
        ck.ok(abs(w.sum() - 1) < 1e-12, "weights do not sum to 1")
        # scale independence (§3.1): doubling T stretches the profile exactly (pairs of bins of 2T == bins of T)
        w2 = edm2_power_weights(g, 2 * T)
        stretch = np.abs(w2[0::2] + w2[1::2] - w).max()
        ck.ok(stretch < 1e-14, f"profile is not scale independent (max diff {stretch:.1e})")
        ck.note(f"sigma_rel {s:.2f}: incremental == closed form (rel err {err:.1e}); T->2T stretch diff {stretch:.1e}")


@register("edm2_posthoc_profile_fit", "EDM2 §3.2, App. C.5 Eq. (134)-(151), Alg. 3; 'nearly perfect reconstruction for "
                                      "sigma_rel in [0.015, 0.250]' with 160-512 snapshots x {0.05, 0.10}")
def t_edm2_posthoc_profile_fit(ck):
    targets = [0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]
    g_in = [edm2_std_to_exp(0.05), edm2_std_to_exp(0.10)]
    table = {}
    for nsnap in (8, 32, 160):
        ts = np.arange(1, nsnap + 1, dtype=np.float64) / nsnap        # snapshot times, T = 1 (units cancel)
        t_i = np.repeat(ts, 2)
        g_i = np.tile(g_in, nsnap)
        res, sums = [], []
        for s in targets:
            g_r = edm2_std_to_exp(s)
            X, A, B = edm2_solve(t_i, g_i, 1.0, g_r)
            x = X[:, 0]
            prr = float(edm2_pdp(np.float64(1.0), g_r, np.float64(1.0), g_r))
            r2 = prr - 2 * float(B[:, 0] @ x) + float(x @ A @ x)          # ||p* - p_r||^2, Eq. (134)
            res.append(math.sqrt(max(r2, 0.0) / prr))
            sums.append(float(x.sum()))
        table[nsnap] = (res, sums)
        ck.note(f"{nsnap:3d} snapshots x 2 EMAs: rel L2 profile error " +
                " ".join(f"{s:.3f}:{r:.1e}" for s, r in zip(targets, res)))
    r160, s160 = table[160]
    ck.ok(max(r160) < 0.01, f"160 snapshots: max profile error {max(r160):.3%} >= 1%")
    ck.ok(max(abs(v - 1) for v in s160) < 0.01, "160 snapshots: reconstruction weights do not sum to ~1")
    for j, s in enumerate(targets):
        if s in (0.05, 0.10):
            continue  # exactly representable: error 0 for every snapshot count
        ck.ok(table[8][0][j] > table[32][0][j] > table[160][0][j], f"error not decreasing with snapshots at {s}")


@register("edm2_posthoc_trajectory", "EDM2 §3.2 + App. C.5 (reconstruction of an unseen EMA length from snapshots of "
                                     "two tracked power EMAs), synthetic trajectory")
def t_edm2_posthoc_trajectory(ck):
    rng = np.random.default_rng(1)
    T, D, S = 1000, 64, 25
    target = rng.standard_normal(D)
    theta = np.zeros((T + 1, D))
    for t in range(1, T + 1):  # SGD-like: drift towards a target + noise
        theta[t] = theta[t - 1] + 0.01 * (target - theta[t - 1]) + 0.05 * rng.standard_normal(D)
    g_in = [edm2_std_to_exp(0.05), edm2_std_to_exp(0.10)]
    outs = [0.03, 0.07, 0.15, 0.20]
    g_out = [edm2_std_to_exp(s) for s in outs]
    emas, direct = np.zeros((2, D)), np.zeros((len(outs), D))
    snaps, snap_t, snap_g = [], [], []
    for t in range(1, T + 1):
        for k, g in enumerate(g_in):
            b = (1 - 1 / t) ** (g + 1)
            emas[k] = b * emas[k] + (1 - b) * theta[t]
        for k, g in enumerate(g_out):
            b = (1 - 1 / t) ** (g + 1)
            direct[k] = b * direct[k] + (1 - b) * theta[t]
        if t % S == 0:
            for k, g in enumerate(g_in):
                snaps.append(emas[k].copy())
                snap_t.append(t)
                snap_g.append(g)
    X, _, _ = edm2_solve(snap_t, snap_g, T, g_out)
    rec = X.T @ np.array(snaps)
    for k, s in enumerate(outs):
        err = np.linalg.norm(rec[k] - direct[k]) / np.linalg.norm(direct[k] - theta[T])
        ck.ok(err < 0.01, f"sigma_rel {s}: reconstruction error {err:.2%} of |EMA - theta_T|")
        ck.note(f"sigma_rel {s:.2f} from {len(snaps)} snapshots: error = {err:.2e} x |EMA - theta_T|")


@register("edm2_ema_mapping", "[pemetaan] EDM2 App. C.1-C.2 Eq. (105)-(111): sigma_rel of our fixed-beta EMA "
                              "(G56CT_EMA, EMA_EXTRA) at our step caps; LS fit of power profiles from final-only EMAs")
def t_edm2_ema_mapping(ck):
    caps = (300, 450, 880, 1200, 1700)
    for beta in (0.98, 0.99, 0.995, 0.998):
        row = []
        for T in caps:
            p = traditional_ema_profile(beta, T)
            row.append(f"T{T}:{profile_sigma_rel(p):.3f}(init {p[0]:.3f})")
        ck.note(f"beta {beta}: sigma_rel " + " ".join(row))
    p300, p1700 = traditional_ema_profile(0.99, 300), traditional_ema_profile(0.99, 1700)
    ck.ok(abs(profile_sigma_rel(p300) - 0.2781) < 5e-4, "beta 0.99 @300: sigma_rel != 0.278")
    ck.ok(abs(profile_sigma_rel(p1700) - 0.0585) < 5e-4, "beta 0.99 @1700: sigma_rel != 0.0585")
    ck.ok(abs(p300[0] - 0.99 ** 300) < 1e-15 and abs(p300[0] - 0.049) < 1e-3, "init mass at 300 steps != 4.9%")
    # untruncated exponential std = sqrt(beta)/(1-beta) steps (the survey's +-33% / +-6% used this approximation)
    ck.note(f"untruncated sqrt(b)/(1-b)/T: T300 {math.sqrt(.99) / .01 / 300:.3f}, T1700 {math.sqrt(.99) / .01 / 1700:.4f}")
    # EMA_EXTRA as queued (0.98,0.995,0.998,0) + the main 0.99: how well can a LINEAR combination of the FINAL-TIME
    # profiles emulate a power profile (EDM2 post-hoc EMA without snapshots)?
    for T in (300, 450, 880):
        basis = np.stack([np.eye(T + 1)[T]] + [traditional_ema_profile(b, T) for b in (0.98, 0.99, 0.995, 0.998)], 1)
        row = []
        for s in (0.05, 0.10, 0.15):
            g = edm2_std_to_exp(s)
            pr = np.concatenate([[0.0], edm2_power_weights(g, T)])
            x, *_ = np.linalg.lstsq(basis, pr, rcond=None)
            res = np.linalg.norm(basis @ x - pr) / np.linalg.norm(pr)
            single = min(np.linalg.norm(basis[:, k] - pr) / np.linalg.norm(pr) for k in range(basis.shape[1]))
            ck.ok(res <= single + 1e-12, "LS fit worse than the best single profile")
            row.append(f"s{s:.2f}: LS {res:.3f} (best single {single:.3f}, max|x| {np.abs(x).max():.1f})")
        ck.note(f"T{T} final-only basis {{raw,.98,.99,.995,.998}}: " + "; ".join(row))


@register("edm2_lora_factor_ema", "[pemetaan] EDM2 Eq. (1)/(105) applied to LoRA: EMA(B)EMA(A) = EMA(BA) - Cov_p(B,A); "
                                  "power-EMA of dW from raw (B,A) snapshots (survey E2)")
def t_edm2_lora_factor_ema(ck):
    g = torch.Generator().manual_seed(0)
    n_in = n_out = 64
    r, T = 4, 450
    Wt = (torch.randn(n_out, r, generator=g) @ torch.randn(r, n_in, generator=g)) / math.sqrt(n_in) * 0.5
    A = (torch.randn(r, n_in, generator=g) / math.sqrt(n_in)).requires_grad_(True)   # our G56CT_INIT=gauss (Init[1])
    B = torch.zeros(n_out, r, requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=3e-3, betas=(0.9, 0.999), weight_decay=0.0)
    Bs, As = [B.detach().clone()], [A.detach().clone()]
    for _ in range(T):
        x = torch.randn(4, n_in, generator=g)
        y = x @ Wt.T + 0.3 * torch.randn(4, n_out, generator=g)
        loss = ((x @ A.T @ B.T - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        Bs.append(B.detach().clone())
        As.append(A.detach().clone())
    Bs, As = torch.stack(Bs).double(), torch.stack(As).double()
    dW = torch.einsum("tor,tri->toi", Bs, As)
    ck.note(f"toy LoRA 64x64 r4, AdamW lr 3e-3, {T} steps: |A_T - A_0|/|A_0| = "
            f"{float((As[-1] - As[0]).norm() / As[0].norm()):.2f} (A moves as much as its init)")
    for s in (0.05, 0.10, 0.20):
        gm = edm2_std_to_exp(s)
        w = torch.tensor(edm2_power_weights(gm, T))
        exact = torch.einsum("t,toi->oi", w, dW[1:])
        eB, eA = torch.einsum("t,tor->or", w, Bs[1:]), torch.einsum("t,tri->ri", w, As[1:])
        cov = torch.einsum("t,toi->oi", w, torch.einsum("tor,tri->toi", Bs[1:] - eB, As[1:] - eA))
        ck.ok(float((exact - eB @ eA - cov).abs().max()) < 1e-12, "Cov identity failed")
        dev = float((eB @ eA - exact).norm() / exact.norm())
        errs = {}
        for S in (T // 32, T // 8):   # raw snapshots, bin (t_{k-1}, t_k] weight assigned to snapshot t_k
            ks = np.arange(S, T + 1, S)
            if ks[-1] != T:
                ks = np.append(ks, T)
            prev = np.concatenate([[0], ks[:-1]])
            wk = torch.tensor((ks.astype(np.float64) ** (gm + 1) - prev.astype(np.float64) ** (gm + 1)) / T ** (gm + 1))
            rec = torch.einsum("k,koi->oi", wk, dW[torch.tensor(ks)])
            errs[S] = float((rec - exact).norm() / exact.norm())
        ck.ok(errs[T // 32] < errs[T // 8] and errs[T // 32] < 0.03, f"snapshot reconstruction not accurate at {s}")
        ck.note(f"power sigma_rel {s:.2f}: factor-EMA vs dW-EMA dev {dev:.4f}; raw snapshots every {T // 32} steps "
                f"err {errs[T // 32]:.4f}, every {T // 8} steps err {errs[T // 8]:.4f}; "
                f"|EMA - final| {float((exact - dW[-1]).norm() / exact.norm()):.3f}")
    # our actual default: traditional beta 0.99, EMA initialised to the initial factors
    beta = 0.99
    eB, eA, eW = Bs[0].clone(), As[0].clone(), dW[0].clone()
    for t in range(1, T + 1):
        eB = beta * eB + (1 - beta) * Bs[t]
        eA = beta * eA + (1 - beta) * As[t]
        eW = beta * eW + (1 - beta) * dW[t]
    ck.note(f"G56CT_EMA=0.99 @T{T}: |EMA(B)EMA(A) - EMA(BA)| / |EMA(BA)| = {float((eB @ eA - eW).norm() / eW.norm()):.4f}, "
            f"|EMA(B)EMA(A)| / |EMA(BA)| = {float((eB @ eA).norm() / eW.norm()):.4f}")


# ======================================================================================================================
# Model Stock (Jang et al., ECCV 2024)
# ======================================================================================================================
def ms_t(N, cos):
    """§3 Eq. (3) (= App. D Eq. (12)); N = 2 gives Eq. (2): t = 2cos/(1+cos)."""
    return N * cos / (1 + (N - 1) * cos)


@register("ms_formula_geometry", "Model Stock §3 Eq. (2)-(3), App. B (thin shell, orthogonality) and App. D Eq. (8)-(12)")
def t_ms_formula_geometry(ck):
    rng = np.random.default_rng(0)
    dim, l = 64, 1.7
    for c in (0.1, 0.3, 0.6, 0.9):
        for N in (2, 3, 5):
            Q, _ = np.linalg.qr(rng.standard_normal((dim, N + 1)))
            mu = Q[:, 0] * l * math.sqrt(c)                 # center; w0 = pre-trained = origin
            W = mu[:, None] + Q[:, 1:] * l * math.sqrt(1 - c)  # Eq. (1): |w_i| = l, w_i.w_j = l^2 cos
            wavg = W.mean(1)
            cos = float(W[:, 0] @ W[:, 1]) / (np.linalg.norm(W[:, 0]) * np.linalg.norm(W[:, 1]))
            t_proj = float(mu @ wavg / (wavg @ wavg))        # App. D Eq. (9)
            ck.ok(abs(cos - c) < 1e-12 and abs(t_proj - ms_t(N, cos)) < 1e-12, f"t formula failed c={c} N={N}")
            coef, *_ = np.linalg.lstsq(W, mu, rcond=None)    # closest point to mu in span(w0=0, w1..wN)
            ck.ok(np.linalg.norm(W @ coef - ms_t(N, cos) * wavg) < 1e-12, f"w_H is not the projection c={c} N={N}")
            ck.ok(abs(float(mu @ (W[:, 0] - mu))) < 1e-12, "App. B property (w0-mu) _|_ (w_i-mu) violated")
    ck.note("t = N cos/(1+(N-1)cos): " + ", ".join(f"cos {c}: N2 {ms_t(2, c):.3f} N3 {ms_t(3, c):.3f}"
                                                   for c in (0.05, 0.1, 0.3, 0.6, 0.9)))
    ck.note(f"our R1 seed deltas (STRATEGI 24 Sep: cos 0.104-0.109, per-module median 0.02-0.07) -> "
            f"t(N=2) = {ms_t(2, 0.109):.3f} .. {ms_t(2, 0.02):.3f}")


@register("ms_table1_distances", "Model Stock §2.2 Table 1 (|w - mu| of 1/2/3/5-averages, pseudo-center = mean of 50) "
                                 "and Table 7 (uniform-average column)")
def t_ms_table1(ck):
    d1 = 13.133  # Table 1, single fine-tuned model
    paper = {2: 9.192, 3: 7.439, 5: 5.633}
    for N, v in paper.items():
        pred = d1 * math.sqrt((1 / N - 1 / 50) / (1 - 1 / 50))  # orthogonal equal-norm noise, members inside the 50
        rel = (pred - v) / v
        ck.ok(abs(rel) < 0.005, f"N={N}: predicted {pred:.3f} vs Table 1 {v} ({rel:+.2%})")
        ck.note(f"N={N}: thin-shell prediction {pred:.3f} vs Table 1 {v} ({rel:+.2%})")
    rng = np.random.default_rng(3)  # Monte Carlo check of the sqrt((1/N-1/50)/(1-1/50)) factor
    dim = 20000
    mu = rng.standard_normal(dim) * 0.5
    Wk = mu[:, None] + rng.standard_normal((dim, 50))
    pc = Wk.mean(1)
    d_single = np.linalg.norm(Wk[:, 0] - pc)
    for N in (2, 3, 5):
        mc = np.linalg.norm(Wk[:, :N].mean(1) - pc) / d_single
        th = math.sqrt((1 / N - 1 / 50) / (1 - 1 / 50))
        ck.ok(abs(mc - th) < 0.01, f"MC factor N={N}: {mc:.4f} vs {th:.4f}")
    ck.note(f"INFO Table 7 uniform N=4 prints 5.63 but the same model predicts {d1 * math.sqrt((0.25 - 0.02) / 0.98):.2f} "
            f"(5.63 equals Table 1's N=5 value) -> probable typo in the paper, not verifiable")


@register("ms_vs_uniform_average", "Model Stock §2.1 assumptions + §3 Eq. (3): closer to the center than the uniform "
                                   "average; [batas asumsi] anisotropic loss; [pemetaan] LoRA cos from factors")
def t_ms_vs_average(ck):
    rng = np.random.default_rng(1)
    dim = 4096
    for c_true in (0.1, 0.5):
        s2 = (1 - c_true) / c_true          # |mu| = 1, per-model noise norm^2 = s2  ->  cos = 1/(1+s2)
        acc = {N: [] for N in (2, 3, 5)}
        for _ in range(10):
            mu = rng.standard_normal(dim)
            mu /= np.linalg.norm(mu)
            for N in acc:
                Wm = mu[:, None] + rng.standard_normal((dim, N)) * math.sqrt(s2 / dim)
                Wn = Wm / np.linalg.norm(Wm, axis=0)
                cosm = float(np.mean([Wn[:, i] @ Wn[:, j] for i in range(N) for j in range(i + 1, N)]))
                wavg = Wm.mean(1)
                acc[N].append((np.linalg.norm(wavg - mu), np.linalg.norm(ms_t(N, cosm) * wavg - mu)))
        for N, v in acc.items():
            a, m = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
            th = math.sqrt((1 - c_true) / (1 + (N - 1) * c_true))   # |w_H - mu| / |mu| under the assumptions
            ck.ok(m < a and abs(m - th) < 0.02, f"c={c_true} N={N}: MS {m:.3f} avg {a:.3f} theory {th:.3f}")
            ck.note(f"isotropic, cos {c_true}, N={N}: |avg-mu| {a:.3f}  |MS-mu| {m:.3f} (theory {th:.3f}, "
                    f"t={ms_t(N, c_true):.3f})")
    # [batas asumsi]: same geometry, but the loss is flat along the seed-specific directions
    c, N = 0.1, 2
    k = 64
    mu = np.zeros(2 * k)
    mu[:k] = rng.standard_normal(k)
    mu *= 1 / np.linalg.norm(mu)
    noise = np.zeros((2 * k, N))
    noise[k:] = rng.standard_normal((k, N))
    noise *= math.sqrt((1 - c) / c) / np.linalg.norm(noise, axis=0)
    W = mu[:, None] + noise
    wavg, wms = W.mean(1), ms_t(N, c) * W.mean(1)
    h = np.concatenate([np.ones(k), np.full(k, 0.01)])     # curvature: stiff shared part, flat seed part
    loss = lambda w: float(((w - mu) ** 2 * h).sum())       # noqa: E731
    iso = lambda w: float(((w - mu) ** 2).sum())            # noqa: E731
    ck.ok(iso(wms) < iso(wavg), "isotropic loss should prefer Model Stock")
    ck.ok(loss(wavg) < loss(wms), "flat-noise loss should prefer the plain average")
    ck.note(f"cos 0.1 N=2: isotropic loss avg {iso(wavg):.3f} vs MS {iso(wms):.3f}; seed noise in flat directions "
            f"(curvature 0.01): avg {loss(wavg):.3f} vs MS {loss(wms):.3f} -> t={ms_t(N, c):.3f} is only right under Obs. 2")
    # [pemetaan]: cos between LoRA deltas without forming dW: <B1A1, B2A2>_F = tr((B1^T B2)(A2 A1^T))
    g = np.random.default_rng(5)
    B1, A1, B2, A2 = g.standard_normal((300, 16)), g.standard_normal((16, 200)), g.standard_normal((300, 16)), \
        g.standard_normal((16, 200))
    ip = lambda Ba, Aa, Bb, Ab: float(np.trace((Ba.T @ Bb) @ (Ab @ Aa.T)))  # noqa: E731
    cos_f = ip(B1, A1, B2, A2) / math.sqrt(ip(B1, A1, B1, A1) * ip(B2, A2, B2, A2))
    D1, D2 = B1 @ A1, B2 @ A2
    cos_d = float((D1 * D2).sum() / (np.linalg.norm(D1) * np.linalg.norm(D2)))
    ck.ok(abs(cos_f - cos_d) < 1e-12, "factor-space cosine != dense cosine")
    ck.note(f"LoRA dW cosine from r x r factor products == dense ({cos_f:.6f})")


# ======================================================================================================================
# LoRA+ (Hayou, Ghosh, Yu, ICML 2024)
# ======================================================================================================================
def loraplus_toy_deltas(n, eta_a, eta_b, steps, s0, y=1.0):
    """§3.1 Eq. (2) toy f(x) = (W* + b a^T) x with W* = 0, loss (f-y)^2/2, Init[1] (b=0, a~N(0,1/n)), plain GD.
    The GD dynamics depend on (a, x) only through s = a^T x and |x|^2:  s <- s - eta_a b U |x|^2,  b <- b - eta_b s U,
    so they are simulated EXACTLY with scalars for x = +-1 entries (|x|^2 = n, x = Theta(1)) and s_0 = a_0^T x ~ N(0, 1)
    (its exact law for a_0 ~ N(0, I/n)); s0 is a vector of independent draws. Returns per step
    (|delta^1_t|, |delta^2_t|) = (|b_{t-1} (s_t - s_{t-1})|, |(b_t - b_{t-1}) s_{t-1}|) as arrays over the draws."""
    b = np.zeros_like(s0)
    s = s0.copy()
    out = []
    for _ in range(steps):
        U = b * s - y
        s_new = s - eta_a * b * U * n
        b_new = b - eta_b * s * U
        out.append((np.abs(b * (s_new - s)), np.abs((b_new - b) * s)))
        s, b = s_new, b_new
    return out


def loglog_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


@register("loraplus_toy_linear_gd", "LoRA+ §3.1 Eq. (2), Prop. 1 (equal LR is inefficient) and Prop. 2 (eta_a=Theta(1/n), "
                                    "eta_b=Theta(1)); proofs App. A.3-A.4")
def t_loraplus_toy_linear(ck):
    ns = [2 ** k for k in (8, 10, 12, 14, 16)]
    s0 = np.random.default_rng(0).standard_normal(4000)
    t_meas = 2  # delta^i_t at t=3 (index 2): both parts are non-zero for t > 1
    cases = {"LoRA+ eta_a=0.5/n, eta_b=0.5": (lambda n: 0.5 / n, lambda n: 0.5),
             "equal eta=0.5/sqrt(n)": (lambda n: 0.5 / math.sqrt(n), lambda n: 0.5 / math.sqrt(n)),
             "equal eta=0.5/n": (lambda n: 0.5 / n, lambda n: 0.5 / n),
             "equal eta=0.02 (Theta(1))": (lambda n: 0.02, lambda n: 0.02)}
    slopes = {}
    for name, (fa, fb) in cases.items():
        d1, d2 = [], []
        for n in ns:
            v1, v2 = loraplus_toy_deltas(n, fa(n), fb(n), 3, s0)[t_meas]
            d1.append(float(np.median(v1)))
            d2.append(float(np.median(v2)))
        slopes[name] = (loglog_slope(ns, d1), loglog_slope(ns, d2))
        ck.note(f"{name:28s}: slope(delta1) {slopes[name][0]:+.2f}  slope(delta2) {slopes[name][1]:+.2f}  "
                f"(median over 4000 draws, n = {ns[0]}..{ns[-1]})")
    s = slopes["LoRA+ eta_a=0.5/n, eta_b=0.5"]
    ck.ok(abs(s[0]) < 0.1 and abs(s[1]) < 0.1, f"LoRA+ scaling should keep both deltas Theta(1): {s}")
    s = slopes["equal eta=0.5/sqrt(n)"]
    ck.ok(-0.6 < s[0] < -0.4 and -0.6 < s[1] < -0.4, f"equal eta ~ n^-1/2: both deltas should decay ~n^-1/2: {s}")
    s = slopes["equal eta=0.5/n"]
    ck.ok(s[0] < -0.8 and s[1] < -0.8, f"equal eta ~ 1/n: deltas should vanish: {s}")
    s = slopes["equal eta=0.02 (Theta(1))"]
    ck.ok(max(s) > 0.5, f"equal Theta(1) eta: at least one delta should blow up with width: {s}")


def loraplus_adam_deltas(n, lr_a, lr_b, r=4, steps=2, seed=0):
    """§4 setting: one LoRA layer Z_bar = W* Z + B A Z, single sample, Adam, Init[1]. Returns rms(delta^1_t),
    rms(delta^2_t) at the last step (delta^1 = B_{t-1} dA_t Z, delta^2 = dB_t A_{t-1} Z, §4) and rms(Z_A) = rms(A_t Z)
    (Def. 3: stability needs Z_A = O(1))."""
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, generator=g)
    Wst = torch.randn(n, n, generator=g) / math.sqrt(n)
    tgt = torch.randn(n, generator=g)
    A = (torch.randn(r, n, generator=g) / math.sqrt(n)).requires_grad_(True)
    B = torch.zeros(n, r, requires_grad=True)
    opt = torch.optim.Adam([{"params": [A], "lr": lr_a}, {"params": [B], "lr": lr_b}], betas=(0.9, 0.999), eps=1e-12)
    d = None
    for _ in range(steps):
        A0, B0 = A.detach().clone(), B.detach().clone()
        loss = 0.5 * ((Wst @ Z + B @ (A @ Z) - tgt) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        d1 = B0 @ ((A.detach() - A0) @ Z)
        d2 = (B.detach() - B0) @ (A0 @ Z)
        d = (float(d1.pow(2).mean().sqrt()), float(d2.pow(2).mean().sqrt()), float((A.detach() @ Z).pow(2).mean().sqrt()))
    return d


@register("loraplus_adam_width", "LoRA+ §4 Def. 2-5 (LoRA features, stability, efficiency), Thm. 1 (Adam: eta_A = Theta(1/n), "
                                 "eta_B = Theta(1) is efficient; eta_A = eta_B is not), proof App. A.5")
def t_loraplus_adam(ck):
    ns = [128, 256, 512, 1024]
    cases = {"LoRA+ lr_A=1/n, lr_B=0.05": (lambda n: 1.0 / n, lambda n: 0.05),
             "equal lr=1/n": (lambda n: 1.0 / n, lambda n: 1.0 / n),
             "equal lr=0.05": (lambda n: 0.05, lambda n: 0.05)}
    slopes = {}
    for name, (fa, fb) in cases.items():
        d1, d2, za = [], [], []
        for n in ns:
            v = [loraplus_adam_deltas(n, fa(n), fb(n), seed=s) for s in range(3)]
            d1.append(float(np.exp(np.mean([math.log(x[0]) for x in v]))))
            d2.append(float(np.exp(np.mean([math.log(x[1]) for x in v]))))
            za.append(float(np.exp(np.mean([math.log(x[2]) for x in v]))))
        slopes[name] = (loglog_slope(ns, d1), loglog_slope(ns, d2), loglog_slope(ns, za))
        ck.note(f"{name:26s}: slope(delta1) {slopes[name][0]:+.2f}  slope(delta2) {slopes[name][1]:+.2f}  "
                f"slope(Z_A) {slopes[name][2]:+.2f}  (at n={ns[0]}: {d1[0]:.2e}/{d2[0]:.2e}/{za[0]:.2e})")
    s = slopes["LoRA+ lr_A=1/n, lr_B=0.05"]
    ck.ok(max(abs(v) for v in s) < 0.2, f"LoRA+ Adam scaling should keep delta1, delta2 and Z_A all Theta(1): {s}")
    s = slopes["equal lr=1/n"]
    ck.ok(s[0] < -0.7 and s[1] < -0.7, f"equal lr ~ 1/n: both feature updates should vanish: {s}")
    s = slopes["equal lr=0.05"]
    ck.ok(s[0] > 0.7 and s[2] > 0.7, f"equal Theta(1) lr: delta1 and Z_A should blow up ~n (unstable): {s}")


def loraplus_mlp_losses(seed, etas, steps=200, d=5, n=100, r=4, ntr=1000, nte=100):
    """§3.2 Eq. (3) / App. C.1.1: f(x) = W_out relu(B A relu(W_in x)), X ~ N(0, I_5), Y = sin(mean(X)), W_in ~ N(0,1),
    W_out ~ N(0,1/n), A ~ N(0,1/n), B ~ N(0,1); only A, B trained, full-batch GD, 200 steps. Batched over (eta_A, eta_B),
    manual gradients of L = mean((f - y)^2). Returns (train, test) losses per grid point (1e9 = diverged)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(ntr + nte, d, generator=g)
    Y = torch.sin(X.mean(1))
    Win = torch.randn(n, d, generator=g)
    Wout = torch.randn(n, generator=g) / math.sqrt(n)
    A0 = torch.randn(r, n, generator=g) / math.sqrt(n)
    B0 = torch.randn(n, r, generator=g)
    H = torch.relu(X @ Win.T)
    Htr, Hte, Ytr, Yte = H[:ntr], H[ntr:], Y[:ntr], Y[ntr:]
    G = len(etas)
    ea = torch.tensor([e[0] for e in etas], dtype=torch.float32)[:, None, None]
    eb = torch.tensor([e[1] for e in etas], dtype=torch.float32)[:, None, None]
    A, B = A0.expand(G, r, n).clone(), B0.expand(G, n, r).clone()
    for _ in range(steps):
        ZA = torch.matmul(Htr, A.transpose(1, 2))
        ZB = torch.bmm(ZA, B.transpose(1, 2))
        f = torch.relu(ZB) @ Wout
        dZB = ((2.0 / ntr) * (f - Ytr))[..., None] * Wout * (ZB > 0)
        gB = torch.bmm(dZB.transpose(1, 2), ZA)
        gA = torch.matmul(torch.bmm(dZB, B).transpose(1, 2), Htr)
        A, B = A - ea * gA, B - eb * gB

    def loss(Hs, Ys):
        f = torch.relu(torch.bmm(torch.matmul(Hs, A.transpose(1, 2)), B.transpose(1, 2))) @ Wout
        return torch.nan_to_num(((f - Ys) ** 2).mean(1), nan=1e9, posinf=1e9).clamp(max=1e9).numpy()
    return loss(Htr, Ytr), loss(Hte, Yte)


@register("loraplus_toy_mlp_grid", "LoRA+ §3.2 Eq. (3), Fig. 2 (3-seed average; best (eta_A, eta_B) has eta_B >> eta_A), "
                                   "setup App. C.1.1")
def t_loraplus_mlp(ck):
    ga = [1e-5, 1e-4, 1e-3, 1e-2]
    gb = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10 ** 0.5]
    grid = [(a, b) for a in ga for b in gb]
    edges = [(10 ** -1.5, 1e-5), (1e-1, 1e-5), (1e-5, 1e1)]   # stability probes
    etas = grid + edges
    runs = [loraplus_mlp_losses(s, etas) for s in range(3)]
    te = np.mean([r[1] for r in runs], 0)
    tr = np.mean([r[0] for r in runs], 0)
    G = len(grid)
    i = int(np.argmin(te[:G]))
    diag = [k for k, (a, b) in enumerate(grid) if a == b]
    kd = min(diag, key=lambda k: te[k])
    ratio = grid[i][1] / grid[i][0]
    ck.note(f"3-seed mean test MSE: best (eta_A, eta_B) = ({grid[i][0]:.0e}, {grid[i][1]:.0e}) -> {te[i]:.4f}, "
            f"eta_B/eta_A = {ratio:.0f}; best on eta_A = eta_B: ({grid[kd][0]:.0e}) -> {te[kd]:.4f}")
    it = int(np.argmin(tr[:G]))
    ck.note(f"3-seed mean train MSE: best ({grid[it][0]:.0e}, {grid[it][1]:.0e}), eta_B/eta_A = {grid[it][1] / grid[it][0]:.0f}")
    ck.ok(ratio >= 10, f"best test point should have eta_B >> eta_A (ratio {ratio})")
    ck.ok(te[kd] >= 2 * te[i], f"best equal-LR test loss {te[kd]:.4f} should be clearly worse than best {te[i]:.4f}")
    for s, (ltr, lte) in enumerate(runs):
        j = int(np.argmin(lte[:G]))
        stab_a = max([a for (a, b), v in zip(etas, lte) if b == 1e-5 and v < 1e8])
        stab_b = max([b for (a, b), v in zip(etas, lte) if a == 1e-5 and v < 1e8])
        ck.note(f"INFO seed {s}: own best ({grid[j][0]:.0e}, {grid[j][1]:.0e}) ratio {grid[j][1] / grid[j][0]:g}; largest "
                f"stable eta_A (eta_B=1e-5) {stab_a:.1e}, largest stable eta_B (eta_A=1e-5) {stab_b:.1e}")
    ck.note("INFO Fig. 2 caption prints (eta_A*, eta_B*) = (2.78, 1.29e-4); here eta_A >= 1e-1 diverges, so the "
            "printed order looks swapped (not verifiable)")


# ======================================================================================================================
# WiSE-FT (CVPR 2022) + Model soups (ICML 2022)
# ======================================================================================================================
def _mlp_setup(seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    N, din, h, K = 512, 10, 32, 5
    X = torch.randn(N, din, generator=g, dtype=torch.float64)
    P = h * din + h + K * h + K

    def f(th):
        i = 0
        W1 = th[i:i + h * din].view(h, din); i += h * din  # noqa: E702
        b1 = th[i:i + h]; i += h  # noqa: E702
        W2 = th[i:i + K * h].view(K, h); i += K * h  # noqa: E702
        b2 = th[i:i + K]
        return torch.tanh(X @ W1.T + b1) @ W2.T + b2
    th0 = torch.randn(P, generator=g, dtype=torch.float64) * 0.5
    u = torch.randn(P, generator=g, dtype=torch.float64)
    u = u / u.norm() * math.sqrt(P) * 0.05
    Yc = torch.randint(0, K, (N,), generator=g)
    Yr = torch.randn(N, K, generator=g, dtype=torch.float64)
    return f, th0, u, Yc, Yr


@register("soups_jensen_mse", "Soups §4 (soup vs ensemble), WiSE-FT §3 (linear classifier: weight = output ensemble) and "
                              "App. F Prop. 1 Eq. (12)-(17); MSE identity for our velocity-MSE metric")
def t_soups_jensen(ck):
    f, th0, u, _, Yr = _mlp_setup(0)
    mse = lambda out: float(((out - Yr) ** 2).sum(1).mean())   # noqa: E731
    th1 = th0 + u
    Df = f(th1) - f(th0)
    for a in (0.25, 0.5, 0.75):
        ens = (1 - a) * f(th0) + a * f(th1)
        ident = (1 - a) * mse(f(th0)) + a * mse(f(th1)) - a * (1 - a) * float((Df ** 2).sum(1).mean())
        ck.ok(abs(mse(ens) - ident) < 1e-10, f"ensemble MSE identity failed at alpha={a}")
    ck.note(f"ensemble MSE = (1-a)L0 + aL1 - a(1-a)E|f1-f0|^2 exactly (Jensen gap at a=0.5: "
            f"{0.25 * float((Df ** 2).sum(1).mean()):.4f})")
    # model linear in its parameters (a LoRA'd linear layer is linear in dW): weight average == output average
    g = torch.Generator().manual_seed(1)
    Xl = torch.randn(256, 12, generator=g, dtype=torch.float64)
    W0 = torch.randn(7, 12, generator=g, dtype=torch.float64)
    D1, D2 = torch.randn(7, 12, generator=g, dtype=torch.float64), torch.randn(7, 12, generator=g, dtype=torch.float64)
    Yl = torch.randn(256, 7, generator=g, dtype=torch.float64)
    lin = lambda D: Xl @ (W0 + D).T                           # noqa: E731
    lmse = lambda out: float(((out - Yl) ** 2).sum(1).mean())  # noqa: E731
    for a in (0.3, 0.5):
        soup = lin((1 - a) * D1 + a * D2)
        ens = (1 - a) * lin(D1) + a * lin(D2)
        ck.ok(float((soup - ens).abs().max()) < 1e-12, "linear model: soup != ensemble")
        ck.ok(lmse(soup) <= (1 - a) * lmse(lin(D1)) + a * lmse(lin(D2)) + 1e-12, "Jensen violated for linear model")
    ck.note("linear-in-parameters model (WiSE-FT App. F Prop. 1): soup == ensemble, so L(avg) <= avg L")


@register("wiseft_ema_identity", "WiSE-FT App. C.3.2: EMA initialised at theta_0 (our G56CT_EMA) == WiSE-FT with alpha = "
                                 "1 - beta^T applied to the debiased EMA; [pemetaan] consequence for LoRA factors")
def t_wiseft_ema(ck):
    rng = np.random.default_rng(2)
    T, beta = 300, 0.99
    theta = rng.standard_normal((T + 1, 6)).cumsum(0)
    mu_b, mu_0 = theta[0].copy(), np.zeros(6)       # biased (mu_0 = theta_0) and zero-initialised EMA
    for t in range(1, T + 1):
        mu_b = beta * mu_b + (1 - beta) * theta[t]
        mu_0 = beta * mu_0 + (1 - beta) * theta[t]
    deb = mu_0 / (1 - beta ** T)
    alpha = 1 - beta ** T
    ck.ok(np.abs(mu_b - ((1 - alpha) * theta[0] + alpha * deb)).max() < 1e-12, "App. C.3.2 identity failed")
    ck.note(f"biased EMA = WiSE-FT(theta_0, debiased EMA, alpha = 1 - beta^T) exactly; beta 0.99: alpha = "
            + ", ".join(f"T{T_}: {1 - beta ** T_:.4f}" for T_ in (300, 450, 880)))
    # LoRA with G56CT_INIT=gauss: B_0 = 0, A_0 random -> EMA(B) EMA(A) = alpha^2 B_deb A_deb + alpha(1-alpha) B_deb A_0
    g = np.random.default_rng(4)
    B0, A0 = np.zeros((8, 3)), g.standard_normal((3, 10)) / math.sqrt(10)
    Bt = g.standard_normal((T + 1, 8, 3)).cumsum(0) * 0.01
    At = A0 + g.standard_normal((T + 1, 3, 10)).cumsum(0) * 0.01
    Bt[0], At[0] = B0, A0
    eB, eA, zB, zA = B0.copy(), A0.copy(), np.zeros_like(B0), np.zeros_like(A0)
    for t in range(1, T + 1):
        eB, eA = beta * eB + (1 - beta) * Bt[t], beta * eA + (1 - beta) * At[t]
        zB, zA = beta * zB + (1 - beta) * Bt[t], beta * zA + (1 - beta) * At[t]
    Bd, Ad = zB / alpha, zA / alpha
    ck.ok(np.abs(eB @ eA - (alpha ** 2 * Bd @ Ad + alpha * (1 - alpha) * Bd @ A0)).max() < 1e-12, "LoRA factor form failed")
    ck.note("LoRA (gauss init): EMA(B)EMA(A) = a^2 B_deb A_deb + a(1-a) B_deb A_0, a = 1-beta^T -> implicit shrink a^2 = "
            + ", ".join(f"T{T_}: {(1 - beta ** T_) ** 2:.3f}" for T_ in (300, 450, 880)))


@register("soups_eq1_approx", "Soups §4 Eq. (1) (derivation App. K), cross-entropy; MSE analogue derived the same way")
def t_soups_eq1(ck):
    f, th0, u, Yc, Yr = _mlp_setup(0)
    ce = lambda out: torch.nn.functional.cross_entropy(out, Yc)   # noqa: E731
    mse = lambda out: ((out - Yr) ** 2).sum(1).mean()             # noqa: E731
    h = 1e-3
    for eps in (1.0, 0.25):
        th1 = th0 + eps * u
        a = 0.5
        tha = (1 - a) * th0 + a * th1
        d = th1 - th0
        fens = (1 - a) * f(th0) + a * f(th1)
        Df = f(th1) - f(th0)
        # CE: L_soup - L_ens ~ a(1-a)/2 * ( -d2/da2 L_soup + beta^2 E Var_{Y~softmax}[Df_Y] ), beta = 1
        d2 = (ce(f(tha + h * d)) - 2 * ce(f(tha)) + ce(f(tha - h * d))) / h ** 2
        p = torch.softmax(f(tha), 1)
        var = ((p * Df ** 2).sum(1) - (p * Df).sum(1) ** 2).mean()
        exact = float(ce(f(tha)) - ce(fens))
        approx = float(a * (1 - a) / 2 * (-d2 + var))
        rel = abs(approx - exact) / abs(exact)
        ck.ok(rel < 0.02, f"CE Eq. (1) off by {rel:.1%} at eps={eps}")
        # MSE analogue (Hessian of the loss in f is 2I): L_soup - L_ens ~ a(1-a)/2 * ( -d2/da2 L_soup + 2 E|Df|^2 )
        d2m = (mse(f(tha + h * d)) - 2 * mse(f(tha)) + mse(f(tha - h * d))) / h ** 2
        exm = float(mse(f(tha)) - mse(fens))
        apm = float(a * (1 - a) / 2 * (-d2m + 2 * (Df ** 2).sum(1).mean()))
        relm = abs(apm - exm) / abs(exm)
        ck.ok(relm < 0.02, f"MSE analogue off by {relm:.1%} at eps={eps}")
        # Jensen along the path: L_soup - [(1-a)L0 + aL1] ~ -a(1-a)/2 * d2/da2 L_soup (convex path => soup wins)
        lin_int = (1 - a) * float(mse(f(th0))) + a * float(mse(f(th1)))
        ck.note(f"eps {eps}: CE soup-ens exact {exact:.3e} approx {approx:.3e} ({rel:.2%}); MSE exact {exm:.3e} "
                f"approx {apm:.3e} ({relm:.2%}); MSE soup - lin.interp. {float(mse(f(tha))) - lin_int:.3e} vs "
                f"-a(1-a)/2 L'' {float(-a * (1 - a) / 2 * d2m):.3e}")


# ======================================================================================================================
# DiffFit (Xie et al., ICCV 2023)
# ======================================================================================================================
@register("difffit_theorem2_ls", "DiffFit §3.3 Thm. 1 (informal) / App. §9 Thm. 2 Eq. (2)-(4): least squares on the "
                                 "scale factors recovers gamma*; sub-Gaussian norm per §9.2 Def. (= Eq. (33))")
def t_difffit_theorem(ck):
    rng = np.random.default_rng(0)
    D, m = 8, 4000                      # m = Omega(D^2 log D) (D^2 log D = 133)
    W = rng.standard_normal((D, D)) / math.sqrt(D)
    b = rng.standard_normal(D) * 0.5
    xg, wg = np.polynomial.hermite_e.hermegauss(120)
    wg = wg / math.sqrt(2 * math.pi)
    Vmin = min(float(np.sum(wg * np.tanh(b[i] + np.linalg.norm(W[i]) * xg) ** 2) -
                     np.sum(wg * np.tanh(b[i] + np.linalg.norm(W[i]) * xg)) ** 2) for i in range(D))
    ps = np.linspace(1, 60, 600)         # Eq. (33): eta = sup_p p^-1/2 (E|X|^p)^(1/p); Gaussian sup sits at p = 1
    lg = ps / 2 * math.log(2) + np.array([math.lgamma((p + 1) / 2) for p in ps]) - 0.5 * math.log(math.pi)
    vals = np.exp(lg / ps) / np.sqrt(ps)
    ck.ok(int(np.argmax(vals)) == 0 and abs(vals[0] - math.sqrt(2 / math.pi)) < 1e-12, "Gaussian psi2 sup not at p=1")
    gstar = rng.uniform(0.5, 1.5, D)
    rels = []
    for s in (0.01, 0.03, 0.1):
        E = rng.standard_normal((m, D))
        A = np.tanh(E @ W.T + b)                         # a_i = G(eps_i) = f(W eps_i + b)
        y = (A + rng.standard_normal((m, D)) * s) @ gstar  # y = (A + Z) gamma*
        gh, *_ = np.linalg.lstsq(A, y, rcond=None)       # Eq. (3)
        eta = s * math.sqrt(2 / math.pi)
        bound = 4 * math.sqrt(2) * eta / math.sqrt(Vmin)  # Eq. (4)
        rel = float(np.linalg.norm(gh - gstar) / np.linalg.norm(gstar))
        rels.append(rel)
        ck.ok(rel < bound, f"noise {s}: |g^-g*|/|g*| = {rel:.4f} exceeds the Eq. (4) bound {bound:.4f}")
        ck.note(f"noise sd {s}: relative error {rel:.4f} < bound {bound:.3f} (V_min {Vmin:.3f}, eta {eta:.4f})")
    ck.ok(rels[0] < rels[1] < rels[2], "error should grow with the noise level eta")
    # 'a simple gradient descent algorithm seeks an estimate close to gamma*': Adam on gamma from gamma = 1 (DiffFit init)
    E = rng.standard_normal((m, D))
    A = np.tanh(E @ W.T + b)
    y = (A + rng.standard_normal((m, D)) * 0.03) @ gstar
    gh, *_ = np.linalg.lstsq(A, y, rcond=None)
    At, yt = torch.tensor(A), torch.tensor(y)
    gam = torch.ones(D, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([gam], lr=2e-2)
    for _ in range(1500):
        loss = ((At @ gam - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    gd = float(np.linalg.norm(gam.detach().numpy() - gh) / np.linalg.norm(gh))
    ck.ok(gd < 1e-6, f"Adam on gamma does not reach the LS solution ({gd:.1e})")
    ck.note(f"Adam from gamma=1 (DiffFit init) reaches the LS optimum: rel diff {gd:.1e}")


@register("difffit_gain_param", "[pemetaan] DiffFit Alg. 1 (x + gamma*attn(.), gamma init 1) vs our SmallDelta on a "
                                "post-norm gain w (ideogram4 attention_norm2/ffn_norm2): additive delta, Adam step size")
def t_difffit_gain(ck):
    g = torch.Generator().manual_seed(0)
    nrm = torch.randn(3, 16, generator=g, dtype=torch.float64)     # normalised branch output
    for wmag in (0.05, 0.3, 1.0, 3.0):
        w = torch.full((16,), wmag, dtype=torch.float64)
        delta = torch.zeros(16, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=1e-3)
        loss = ((w + delta) * nrm).sum() ** 2 + ((w + delta) * nrm).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        gamma = 1 + delta.detach() / w                                # equivalent DiffFit gamma
        same = float((((w + delta.detach()) * nrm) - (gamma * (w * nrm))).abs().max())
        ck.ok(same < 1e-12, "(w+delta)*n != gamma*(w*n)")
        step = float((gamma - 1).abs().max())
        ck.ok(abs(step - 1e-3 / wmag) < 1e-4 * (1e-3 / wmag), "first Adam step on delta should move gamma by lr/|w|")
        ck.note(f"|w| = {wmag:4.2f}: one Adam step (lr 1e-3) on delta moves the equivalent gamma by {step:.1e} "
                f"= {step / 1e-3:.2f} x the DiffFit step at the same lr")


# ======================================================================================================================
# Bonnaire et al. (NeurIPS 2025): tau_mem grows linearly with n, tau_gen does not (random-features model, §3)
# ======================================================================================================================
@register("bonnaire_rf_timescales", "Bonnaire+ §3 Eq. (7)-(11), Thm. 3.2 (lambda_min of rho_1, tau_mem = psi_p / "
                                    "(Delta_t lambda_min)), Fig. 5 / App. C.2 closed-form gradient flow, App. D")
def t_bonnaire(ck):
    d, psi_p, t = 16, 64, 0.1
    p = psi_p * d
    e, Dt = math.exp(-t), 1 - math.exp(-2 * t)
    xg, wg = np.polynomial.hermite_e.hermegauss(80)
    wg = wg / math.sqrt(2 * math.pi)
    sig = np.tanh                                    # mu_0 = 0 (assumption iii), sigma_x = 1 (Gamma_t = 1)
    mu1 = float(np.sum(wg * xg * sig(xg)))
    a_t, b2 = mu1, Dt * mu1 ** 2                     # Eq. (12) via Stein's lemma
    inner = np.array([np.sum(wg * sig(e * uu + math.sqrt(Dt) * xg)) for uu in xg])
    v2 = float(np.sum(wg * inner ** 2)) - a_t ** 2 * e ** 2           # Eq. (13)
    s2 = float(np.sum(wg * sig(xg) ** 2)) - a_t ** 2 * e ** 2 - v2 - b2  # Eq. (14)
    ck.note(f"t={t}: a_t {a_t:.4f} b_t^2 {b2:.4f} v_t^2 {v2:.4f} s_t^2 {s2:.4f}")
    g = torch.Generator().manual_seed(0)
    W = torch.randn(p, d, generator=g)

    def feats(x, xi):
        return torch.tanh((e * x + math.sqrt(Dt) * xi) @ W.T / math.sqrt(d))
    K = 32768                                         # population statistics (test loss)
    xp, xip = torch.randn(K, d, generator=g), torch.randn(K, d, generator=g)
    Fp = feats(xp, xip)
    Upop, Vpop = ((Fp.T @ Fp) / K).double(), ((Fp.T @ xip) / K).double()
    taus = np.logspace(-1, 6, 81)
    rows = []
    for psi_n in (2, 4, 8, 16):
        n = psi_n * d
        M = 16384 // n                                # noise draws per training point: n*M = 16 p (E_xi of Eq. (11))
        x, xi = torch.randn(n, d, generator=g), torch.randn(n * M, d, generator=g)
        F = feats(x.repeat_interleave(M, 0), xi)
        U, V = ((F.T @ F) / (n * M)).double(), ((F.T @ xi) / (n * M)).double()
        lam, Q = torch.linalg.eigh(U)
        lam = lam.numpy()
        P = (V.T @ Q).numpy()
        Ct, Pt = (Q.T @ Upop @ Q).numpy(), (Vpop.T @ Q).numpy()
        # App. C.2: A(tau)/sqrt(p) = -(1/sqrt(Dt)) V^T U^-1 (I - exp(-2 Dt U tau / psi_p)), A(0) = 0
        Gm = -(1 / math.sqrt(Dt)) * (1 - np.exp(-2 * Dt * np.outer(taus, lam) / psi_p)) / lam
        colsq = np.sum(P * P, 0)
        Ltr = 1 + (Dt / d) * ((Gm ** 2) * lam) @ colsq + (2 * math.sqrt(Dt) / d) * Gm @ colsq    # Eq. (55)
        Lte = np.array([1 + (Dt / d) * np.sum(((P * gg) @ Ct) * (P * gg)) + (2 * math.sqrt(Dt) / d) * np.sum(P * gg * Pt)
                        for gg in Gm])
        Lg = Lte - Ltr
        tau_mem = float(taus[np.argmax(Lg >= 0.5 * Lg[-1])])       # half of the final generalisation gap
        tau_gen = float(taus[np.argmin(Lte)])                      # best test loss (early-stopping point)
        lam_min = s2 + v2 * (1 - math.sqrt(psi_p / psi_n)) ** 2    # Thm. 3.2 left edge of rho_1 (App. D)
        pred = psi_p / (Dt * lam_min)
        rows.append((n, tau_mem, tau_gen, pred, float(Lg[-1]), float(Lte.min())))
        ck.note(f"psi_n {psi_n:2d} (n={n:3d}): tau_mem(50% gap) {tau_mem:9.3g}  predicted psi_p/(Dt lam_min) {pred:9.3g}  "
                f"tau(best test) {tau_gen:6.3g}  L_gen(inf) {Lg[-1]:.3f}  min L_test {Lte.min():.4f}")
    ns = [r[0] for r in rows]
    s_mem, s_gen = loglog_slope(ns, [r[1] for r in rows]), loglog_slope(ns, [r[2] for r in rows])
    ck.note(f"log-log slope vs n: tau_mem {s_mem:.2f} (paper: ~1), tau(best test) {s_gen:.2f} (paper: ~0)")
    ck.ok(0.8 < s_mem < 1.4, f"tau_mem should grow ~linearly with n (slope {s_mem:.2f})")
    ck.ok(s_gen < 0.35, f"tau_gen should be ~independent of n (slope {s_gen:.2f})")
    ck.ok(all(0.5 < r[1] / r[3] < 2.5 for r in rows), "measured tau_mem not within [0.5, 2.5] x Thm. 3.2 prediction")
    ck.ok(all(rows[k][4] > rows[k + 1][4] for k in range(len(rows) - 1)), "L_gen(inf) should shrink as n grows")
    ck.ok(all(rows[k][1] < rows[k + 1][1] for k in range(len(rows) - 1)), "tau_mem should increase with n")


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-k", default="", help="run only tests whose name contains this substring")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="do not print the measured numbers")
    args = ap.parse_args()
    sel = [t for t in REGISTRY if args.k in t[0]]
    if args.list:
        for name, ref, _ in sel:
            print(f"{name:26s} {ref}")
        return 0
    print(f"paper_tests: {len(sel)} tests, torch {torch.__version__}, numpy {np.__version__}, "
          f"threads {torch.get_num_threads()}, cuda_available={torch.cuda.is_available()} (not used)")
    n_fail, t_all = 0, time.time()
    for name, ref, fn in sel:
        ck = Check()
        t0 = time.time()
        try:
            fn(ck)
        except Exception:  # noqa: BLE001
            ck.failed.append("exception:\n" + traceback.format_exc())
        status = "PASS" if not ck.failed else "FAIL"
        n_fail += bool(ck.failed)
        print(f"{status}  {name}  ({time.time() - t0:.1f}s)  [{ref}]", flush=True)
        if not args.quiet:
            for m in ck.info:
                print(f"      {m}")
        for m in ck.failed:
            print(f"      !! {m}")
    print(f"paper_tests: {len(sel) - n_fail}/{len(sel)} passed in {time.time() - t_all:.1f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
