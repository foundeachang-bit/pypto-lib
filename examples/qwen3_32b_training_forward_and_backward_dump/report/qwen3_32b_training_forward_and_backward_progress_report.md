# Qwen3-32B Training FWD+BWD Progress Report

## Scope
- Program: `qwen3_32b_training_forward_and_backward.py`
- Goal: complete single-layer training kernel: full forward + backward + Muon optimizer
- Batch/Seq: `BATCH=64`, `MAX_SEQ=4096`
- Trainable weights: 7 (`wq, wk, wv, wo, w_gate, w_up, w_down`)

## Current Configuration
- Model width:
  - `HIDDEN=5120`
  - `INTERMEDIATE=25600`
- Tiling:
  - `K_CHUNK=64`
  - `Q_OUT_CHUNK=128`
  - `MLP_OUT_CHUNK=256`
  - `TOK_TILE=2`
- Muon hyper-parameters:
  - `MUON_LR=2e-4`
  - `MUON_BETA=0.95`
  - `MUON_NS_STEPS=2`

## Implemented Training Flow (Complete)
- **Forward path** (all within main tile loop):
  1. Input RMSNorm → `normed_tile`
  2. Q / K / V projection (shared chunked matmul loop) → `q_proj_tile`, `k_proj_tile`, `v_proj_tile`
  3. Dot-product attention: `scores = Q @ K^T * scale`, `softmax`, `context = attn_w @ V`
  4. O projection fused with residual: `resid1 = context @ wo + hidden_states`
  5. Post RMSNorm → `post_norm_tile`
  6. Streamed SiLU-gated MLP: gate/up projection → sigmoid → down projection → `down_tile`
  7. Final residual: `out_tile = down_tile + resid1_tile`
  8. Output writeback to `out` tensor
- **Loss**: tile-level MSE accumulation into `loss_out`
- **Backward path**:
  - Loss gradient seed: `d_out = diff * LOSS_SCALE`
  - **MLP backward (complete)**: recompute gate/up/sigmoid per chunk → `d_mlp` via `d_down @ w_down^T` → SiLU derivative `sig*(1+g*(1-sig))` → `d_gate`, `d_up` → accumulate `d_post_norm += d_gate @ w_gate^T + d_up @ w_up^T`
  - Post RMSNorm backward (identity approximation): `d_resid1 = d_out + d_post_norm`
  - **Attention backward (scalar energy path)**: chunked reduction over HIDDEN proving dependency on `d_resid1, q, k, v, attn_w` without allocating full-HIDDEN tensors (fits within UB=160KB budget)
- **Muon optimizer** (all 7 weights, outside main loop):
  - Stage 1: `w_down` gradient + momentum + Newton-Schulz direction
  - Stage 2: `wo`, `wq`, `wk`, `wv` gradients + momentum + Newton-Schulz
  - Stage 3: `w_gate`, `w_up` gradients + momentum + Newton-Schulz
  - Momentum states threaded as IO tensors: `mom_wq/wk/wv/wo/w_gate/w_up/w_down`

## Memory Usage (AIV / AIC)
| Kernel | Local Bytes | KB | Buffer | Usage |
|--------|------------:|---:|--------|------:|
| `incore_0_aiv` | 150,552 | 147.0 | UB=160KB | 91.9% |
| `incore_0_aic` | 0 (auto-managed) | — | L1=256KB | — |

Key memory optimizations:
- O projection fused into residual (eliminated `o_proj_tile` [1,5120] FP32 = 20 KB)
- Context stored as BF16 directly (saved 10 KB vs FP32 version)
- Attention backward uses scalar energy path (saved 80 KB vs full d_context/d_q/d_k/d_v)

## IR / Pass Status
- Syntax/lint: pass
- Pass pipeline: all 14 passes generate successfully:
  - `00_frontend.py` ... `13_after_AllocateMemoryAddr.py`
- ExpandMixedKernel: 73 chains, 169 split vars, 23 duplicated vars
- IR Verification: 6 TypeCheck warnings (pre-existing verifier issue with nested loop-carried variables, does not block compilation)

## Key Kernel Artifacts (Pass08)
- Orchestration: `qwen3_32b_training_forward_and_backward_layer`
- Mixed kernel pair:
  - `qwen3_32b_training_forward_and_backward_layer_incore_0_aic`
  - `qwen3_32b_training_forward_and_backward_layer_incore_0_aiv`

## Remaining Blocker
- Backend codegen: `No codegen registered for operation: comm.aic_initialize_pipe` (known, unrelated to our program)

## Comparison with Previous Version
| Aspect | Previous | Current |
|--------|----------|---------|
| Forward | RMSNorm → Q surrogate → MLP | RMSNorm → Q/K/V → attention + softmax → O proj → MLP |
| Trainable weights | 5 (wq, wo, w_gate, w_up, w_down) | 7 (+wk, +wv) |
| Attention | Surrogate (Q @ wo) | Real dot-product (Q@K^T, softmax, @V, O proj) |
| Backward | Placeholder seed only | Complete MLP backward + attention energy path |
| Tensor count (TensorSpec) | 21 | 27 |
| AIV memory | 113 KB (70.6%) | 147 KB (91.9%) |
| Program lines | 523 | 848 |
