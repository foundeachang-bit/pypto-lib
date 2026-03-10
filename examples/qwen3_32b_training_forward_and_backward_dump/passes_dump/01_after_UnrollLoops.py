# pypto.program: Qwen332BTrainingForwardBackward
import pypto.language as pl

@pl.program
class Qwen332BTrainingForwardBackward:
    @pl.function
    def qwen3_32b_training_forward_and_backward_layer(self, hidden_states: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], target_states: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk: pl.Tensor[[5120, 5120], pl.BFLOAT16], wv: pl.Tensor[[5120, 5120], pl.BFLOAT16], wo: pl.Tensor[[5120, 5120], pl.BFLOAT16], w_gate: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down: pl.Tensor[[25600, 5120], pl.BFLOAT16], mom_wq: pl.Tensor[[5120, 5120], pl.FP32], mom_wk: pl.Tensor[[5120, 5120], pl.FP32], mom_wv: pl.Tensor[[5120, 5120], pl.FP32], mom_wo: pl.Tensor[[5120, 5120], pl.FP32], mom_w_gate: pl.Tensor[[5120, 25600], pl.FP32], mom_w_up: pl.Tensor[[5120, 25600], pl.FP32], mom_w_down: pl.Tensor[[25600, 5120], pl.FP32], grad_wq: pl.Tensor[[5120, 5120], pl.FP32], grad_wk: pl.Tensor[[5120, 5120], pl.FP32], grad_wv: pl.Tensor[[5120, 5120], pl.FP32], grad_wo: pl.Tensor[[5120, 5120], pl.FP32], grad_w_gate: pl.Tensor[[5120, 25600], pl.FP32], grad_w_up: pl.Tensor[[5120, 25600], pl.FP32], grad_w_down: pl.Tensor[[25600, 5120], pl.FP32], out: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], loss_out: pl.Tensor[[1], pl.FP32]) -> tuple[pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[25600, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[25600, 5120], pl.FP32], pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], pl.Tensor[[1], pl.FP32]]:
        with pl.auto_incore():
            grad_wq: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wq, 0.0)
            grad_wk: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wk, 0.0)
            grad_wv: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wv, 0.0)
            grad_wo: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wo, 0.0)
            grad_w_gate: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.mul(grad_w_gate, 0.0)
            grad_w_up: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.mul(grad_w_up, 0.0)
            grad_w_down: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.mul(grad_w_down, 0.0)
            loss_acc: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            loss_acc: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(loss_acc, 0.0)
            tok_blocks: pl.Scalar[pl.INDEX] = 4096 // 2
            for b in pl.parallel(0, 64, 1, chunk=4):
                for p0_idx in pl.range(0, tok_blocks, 1):
                    p0: pl.Scalar[pl.INDEX] = p0_idx * 2
                    sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(sq_sum, 0.0)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        x_chunk: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [2, 64], [b, p0, k0]), target_type=pl.FP32, mode=2)
                        sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum, 0.000195313), 1e-06))
                    normed_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        x_chunk: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [2, 64], [b, p0, k0]), target_type=pl.FP32, mode=2)
                        gamma: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(input_rms_weight, [1, 64], [0, k0])
                        normed: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(normed_tile, pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), [0, k0])
                    q_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    k_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    v_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for ob in pl.range(0, 40, 1):
                        q0: pl.Scalar[pl.INDEX] = ob * 128
                        q_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        q_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(q_acc, 0.0)
                        k_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        k_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(k_acc, 0.0)
                        v_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        v_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(v_acc, 0.0)
                        for kb in pl.range(0, 80, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 64
                            n_chunk: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(normed_tile, [2, 64], [0, k0]), target_type=pl.BFLOAT16, mode=2)
                            wq_c: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wq, [64, 128], [k0, q0])
                            wk_c: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wk, [64, 128], [k0, q0])
                            wv_c: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wv, [64, 128], [k0, q0])
                            q_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(q_acc, pl.tensor.matmul(n_chunk, wq_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                            k_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(k_acc, pl.tensor.matmul(n_chunk, wk_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                            v_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(v_acc, pl.tensor.matmul(n_chunk, wv_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                        q_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile, pl.tensor.cast(q_acc, target_type=pl.BFLOAT16, mode=2), [0, q0])
                        k_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(k_proj_tile, pl.tensor.cast(k_acc, target_type=pl.BFLOAT16, mode=2), [0, q0])
                        v_proj_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(v_proj_tile, pl.tensor.cast(v_acc, target_type=pl.BFLOAT16, mode=2), [0, q0])
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.create([2, 2], dtype=pl.FP32)
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.mul(scores, 0.0)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        q_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [2, 64], [0, k0]), target_type=pl.FP32, mode=2)
                        k_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(k_proj_tile, [2, 64], [0, k0]), target_type=pl.FP32, mode=2)
                        scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.add(scores, pl.tensor.matmul(q_c, k_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.mul(scores, 0.0139754)
                    scores_exp: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.exp(scores)
                    scores_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(scores_exp)
                    attn_w: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.row_expand_mul(scores_exp, pl.tensor.recip(scores_sum))
                    context_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for ob in pl.range(0, 40, 1):
                        o0: pl.Scalar[pl.INDEX] = ob * 128
                        v_c: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(v_proj_tile, [2, 128], [0, o0]), target_type=pl.FP32, mode=2)
                        ctx_c: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.matmul(attn_w, v_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        context_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(context_tile, pl.tensor.cast(ctx_c, target_type=pl.BFLOAT16, mode=2), [0, o0])
                    resid1_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    for ob in pl.range(0, 40, 1):
                        o0: pl.Scalar[pl.INDEX] = ob * 128
                        o_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        o_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(o_acc, 0.0)
                        for kb in pl.range(0, 80, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 64
                            ctx_c: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(context_tile, [2, 64], [0, k0])
                            wo_c: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wo, [64, 128], [k0, o0])
                            o_acc: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(o_acc, pl.tensor.matmul(ctx_c, wo_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        resid: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [2, 128], [b, p0, o0]), target_type=pl.FP32, mode=2)
                        resid1_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc, resid), [0, o0])
                    sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(sq_sum2, 0.0)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        x_chunk: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(resid1_tile, [2, 64], [0, k0])
                        sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum2, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum2, 0.000195313), 1e-06))
                    post_norm_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        x_chunk: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(resid1_tile, [2, 64], [0, k0])
                        gamma: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(post_rms_weight, [1, 64], [0, k0])
                        normed: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms2), gamma)
                        post_norm_tile: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), [0, k0])
                    down_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    down_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(down_tile, 0.0)
                    for mb in pl.range(0, 100, 1):
                        m0: pl.Scalar[pl.INDEX] = mb * 256
                        gate_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        up_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        gate_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(gate_acc, 0.0)
                        up_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(up_acc, 0.0)
                        for kb in pl.range(0, 80, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 64
                            post_chunk: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(post_norm_tile, [2, 64], [0, k0])
                            wg: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate, [64, 256], [k0, m0])
                            wu: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up, [64, 256], [k0, m0])
                            gate_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(gate_acc, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_acc: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(up_acc, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        sigmoid_chunk: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_acc)), 1.0))
                        mlp_chunk: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.mul(pl.tensor.mul(gate_acc, sigmoid_chunk), up_acc), target_type=pl.BFLOAT16, mode=2)
                        for ob in pl.range(0, 40, 1):
                            o0: pl.Scalar[pl.INDEX] = ob * 128
                            down_prev: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.view(down_tile, [2, 128], [0, o0])
                            wd: pl.Tensor[[256, 128], pl.BFLOAT16] = pl.tensor.view(w_down, [256, 128], [m0, o0])
                            down_part: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk, wd, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            down_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(down_tile, down_part, [0, o0])
                    out_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    for ob in pl.range(0, 40, 1):
                        o0: pl.Scalar[pl.INDEX] = ob * 128
                        out_chunk: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(pl.tensor.view(down_tile, [2, 128], [0, o0]), pl.tensor.view(resid1_tile, [2, 128], [0, o0]))
                        out_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(out_tile, out_chunk, [0, o0])
                        out: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16] = pl.tensor.assemble(out, pl.tensor.cast(out_chunk, target_type=pl.BFLOAT16, mode=2), [b, p0, o0])
                    tgt_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.cast(pl.tensor.view(target_states, [2, 5120], [b, p0, 0]), target_type=pl.FP32, mode=2)
                    diff_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.sub(out_tile, tgt_tile)
                    sq_tile: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(diff_tile, diff_tile)
                    sq_row: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(sq_tile)
                    for ti in pl.range(0, 2, 1):
                        cur: pl.Scalar[pl.FP32] = pl.tensor.read(loss_acc, [0, 0])
                        addv: pl.Scalar[pl.FP32] = pl.tensor.read(sq_row, [ti, 0])
                        acc_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                        acc_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(acc_t, 0.0)
                        acc_t: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(acc_t, cur + addv)
                        loss_acc: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.assemble(loss_acc, acc_t, [0, 0])
                    d_out: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(diff_tile, 0.000390625)
                    d_down: pl.Tensor[[2, 5120], pl.FP32] = d_out
                    d_resid1_bwd: pl.Tensor[[2, 5120], pl.FP32] = d_out
                    d_post_norm: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    d_post_norm: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(d_post_norm, 0.0)
                    for mb in pl.range(0, 100, 1):
                        m0: pl.Scalar[pl.INDEX] = mb * 256
                        gate_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        up_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        gate_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(gate_r, 0.0)
                        up_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(up_r, 0.0)
                        for kb in pl.range(0, 80, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 64
                            post_c: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(post_norm_tile, [2, 64], [0, k0])
                            wg_c: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate, [64, 256], [k0, m0])
                            wu_c: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up, [64, 256], [k0, m0])
                            gate_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(gate_r, pl.tensor.matmul(post_c, wg_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(up_r, pl.tensor.matmul(post_c, wu_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        sig_r: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_r)), 1.0))
                        d_mlp: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        d_mlp: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(d_mlp, 0.0)
                        for ob in pl.range(0, 40, 1):
                            o0: pl.Scalar[pl.INDEX] = ob * 128
                            dd_c: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.view(d_down, [2, 128], [0, o0])
                            wd_c: pl.Tensor[[256, 128], pl.BFLOAT16] = pl.tensor.view(w_down, [256, 128], [m0, o0])
                            d_mlp: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(d_mlp, pl.tensor.matmul(dd_c, wd_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                        one_m_sig: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(sig_r, -1.0), 1.0)
                        silu_deriv: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(sig_r, pl.tensor.add(pl.tensor.mul(gate_r, one_m_sig), 1.0))
                        d_gate: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(pl.tensor.mul(d_mlp, up_r), silu_deriv)
                        d_up: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(d_mlp, pl.tensor.mul(gate_r, sig_r))
                        for kb in pl.range(0, 80, 1):
                            k0: pl.Scalar[pl.INDEX] = kb * 64
                            dpn_old: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(d_post_norm, [2, 64], [0, k0])
                            wg_c: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate, [64, 256], [k0, m0])
                            wu_c: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up, [64, 256], [k0, m0])
                            dpn_new: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(dpn_old, pl.tensor.add(pl.tensor.matmul(d_gate, wg_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32), pl.tensor.matmul(d_up, wu_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)))
                            d_post_norm: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(d_post_norm, dpn_new, [0, k0])
                    d_resid1: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.add(d_resid1_bwd, d_post_norm)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(bwd_energy, 0.0)
                    for kb in pl.range(0, 80, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 64
                        dr_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(d_resid1, [2, 64], [0, k0])
                        q_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile, [2, 64], [0, k0]), target_type=pl.FP32, mode=2)
                        k_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(k_proj_tile, [2, 64], [0, k0]), target_type=pl.FP32, mode=2)
                        v_c: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(v_proj_tile, [2, 64], [0, k0]), target_type=pl.FP32, mode=2)
                        contrib: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(dr_c, pl.tensor.add(pl.tensor.add(q_c, k_c), v_c)))
                        bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy, contrib)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy, pl.tensor.row_sum(attn_w))
                    grad_sink: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(bwd_energy, 0.0)
            proxy_mlp: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_up, [2, 256], [0, 0]), target_type=pl.BFLOAT16, mode=2)
            for qb in pl.range(0, 40, 1):
                q0: pl.Scalar[pl.INDEX] = qb * 128
                proxy_go: pl.Tensor[[2, 128], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(target_states, [2, 128], [0, 0, q0]), target_type=pl.BFLOAT16, mode=2)
                grad_down_raw: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.matmul(proxy_mlp, proxy_go, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_down_prev: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.view(mom_w_down, [256, 128], [0, q0])
                mom_down_new: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_down_prev, 0.95), pl.tensor.mul(grad_down_raw, 0.05))
                muon_down: pl.Tensor[[256, 128], pl.FP32] = mom_down_new
                for _ in pl.range(0, 2, 1):
                    gram: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_down, muon_down, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_down: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_down, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_down, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_w_down: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.assemble(grad_w_down, pl.tensor.mul(muon_down, -0.0002), [0, q0])
                mom_w_down: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.assemble(mom_w_down, mom_down_new, [0, q0])
            proxy_ctx: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(wq, [2, 64], [0, 0])
            proxy_n: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(hidden_states, [2, 64], [0, 0, 0]), target_type=pl.BFLOAT16, mode=2)
            for qb in pl.range(0, 40, 1):
                q0: pl.Scalar[pl.INDEX] = qb * 128
                proxy_tgt: pl.Tensor[[2, 128], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(target_states, [2, 128], [0, 0, q0]), target_type=pl.BFLOAT16, mode=2)
                grad_wo_raw: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_ctx, proxy_tgt, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wo_prev: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wo, [64, 128], [0, q0])
                mom_wo_new: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wo_prev, 0.95), pl.tensor.mul(grad_wo_raw, 0.05))
                muon_wo: pl.Tensor[[64, 128], pl.FP32] = mom_wo_new
                for _ in pl.range(0, 2, 1):
                    gram: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wo, muon_wo, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wo: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wo, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wo, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wo: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wo, pl.tensor.mul(muon_wo, -0.0002), [0, q0])
                mom_wo: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wo, mom_wo_new, [0, q0])
                grad_wq_raw: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n, proxy_tgt, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wq_prev: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wq, [64, 128], [0, q0])
                mom_wq_new: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wq_prev, 0.95), pl.tensor.mul(grad_wq_raw, 0.05))
                muon_wq: pl.Tensor[[64, 128], pl.FP32] = mom_wq_new
                for _ in pl.range(0, 2, 1):
                    gram: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wq, muon_wq, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wq: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wq, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wq, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wq: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wq, pl.tensor.mul(muon_wq, -0.0002), [0, q0])
                mom_wq: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wq, mom_wq_new, [0, q0])
                grad_wk_raw: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n, proxy_tgt, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wk_prev: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wk, [64, 128], [0, q0])
                mom_wk_new: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wk_prev, 0.95), pl.tensor.mul(grad_wk_raw, 0.05))
                muon_wk: pl.Tensor[[64, 128], pl.FP32] = mom_wk_new
                for _ in pl.range(0, 2, 1):
                    gram: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wk, muon_wk, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wk: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wk, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wk, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wk: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wk, pl.tensor.mul(muon_wk, -0.0002), [0, q0])
                mom_wk: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wk, mom_wk_new, [0, q0])
                grad_wv_raw: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n, proxy_tgt, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wv_prev: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wv, [64, 128], [0, q0])
                mom_wv_new: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wv_prev, 0.95), pl.tensor.mul(grad_wv_raw, 0.05))
                muon_wv: pl.Tensor[[64, 128], pl.FP32] = mom_wv_new
                for _ in pl.range(0, 2, 1):
                    gram: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wv, muon_wv, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wv: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wv, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wv, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wv: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wv, pl.tensor.mul(muon_wv, -0.0002), [0, q0])
                mom_wv: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wv, mom_wv_new, [0, q0])
            proxy_post: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(hidden_states, [2, 64], [0, 0, 64]), target_type=pl.BFLOAT16, mode=2)
            for mb in pl.range(0, 100, 1):
                m0: pl.Scalar[pl.INDEX] = mb * 256
                proxy_gg: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_gate, [2, 256], [0, m0]), target_type=pl.BFLOAT16, mode=2)
                proxy_gu: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_up, [2, 256], [0, m0]), target_type=pl.BFLOAT16, mode=2)
                grad_wg_raw: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.matmul(proxy_post, proxy_gg, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                grad_wu_raw: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.matmul(proxy_post, proxy_gu, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wg_prev: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.view(mom_w_gate, [64, 256], [0, m0])
                mom_wu_prev: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.view(mom_w_up, [64, 256], [0, m0])
                mom_wg_new: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wg_prev, 0.95), pl.tensor.mul(grad_wg_raw, 0.05))
                mom_wu_new: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wu_prev, 0.95), pl.tensor.mul(grad_wu_raw, 0.05))
                muon_wg: pl.Tensor[[64, 256], pl.FP32] = mom_wg_new
                muon_wu: pl.Tensor[[64, 256], pl.FP32] = mom_wu_new
                for _ in pl.range(0, 2, 1):
                    gram_wg: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(muon_wg, muon_wg, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram_wu: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(muon_wu, muon_wu, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wg: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wg, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wg, gram_wg, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    muon_wu: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wu, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wu, gram_wu, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_w_gate: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(grad_w_gate, pl.tensor.mul(muon_wg, -0.0002), [0, m0])
                grad_w_up: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(grad_w_up, pl.tensor.mul(muon_wu, -0.0002), [0, m0])
                mom_w_gate: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(mom_w_gate, mom_wg_new, [0, m0])
                mom_w_up: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(mom_w_up, mom_wu_new, [0, m0])
            loss_vec: pl.Tensor[[1], pl.FP32] = pl.tensor.view(loss_acc, [1], [0, 0])
            loss_out: pl.Tensor[[1], pl.FP32] = pl.tensor.assemble(loss_out, loss_vec, [0])
        return grad_wq, grad_wk, grad_wv, grad_wo, grad_w_gate, grad_w_up, grad_w_down, mom_wq, mom_wk, mom_wv, mom_wo, mom_w_gate, mom_w_up, mom_w_down, out, loss_out