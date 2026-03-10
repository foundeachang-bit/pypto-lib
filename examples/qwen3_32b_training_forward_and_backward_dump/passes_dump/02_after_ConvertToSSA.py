# pypto.program: Qwen332BTrainingForwardBackward
import pypto.language as pl

@pl.program
class Qwen332BTrainingForwardBackward:
    @pl.function
    def qwen3_32b_training_forward_and_backward_layer(self, hidden_states_0: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], target_states_0: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], mom_wq_0: pl.Tensor[[5120, 5120], pl.FP32], mom_wk_0: pl.Tensor[[5120, 5120], pl.FP32], mom_wv_0: pl.Tensor[[5120, 5120], pl.FP32], mom_wo_0: pl.Tensor[[5120, 5120], pl.FP32], mom_w_gate_0: pl.Tensor[[5120, 25600], pl.FP32], mom_w_up_0: pl.Tensor[[5120, 25600], pl.FP32], mom_w_down_0: pl.Tensor[[25600, 5120], pl.FP32], grad_wq_0: pl.Tensor[[5120, 5120], pl.FP32], grad_wk_0: pl.Tensor[[5120, 5120], pl.FP32], grad_wv_0: pl.Tensor[[5120, 5120], pl.FP32], grad_wo_0: pl.Tensor[[5120, 5120], pl.FP32], grad_w_gate_0: pl.Tensor[[5120, 25600], pl.FP32], grad_w_up_0: pl.Tensor[[5120, 25600], pl.FP32], grad_w_down_0: pl.Tensor[[25600, 5120], pl.FP32], out_0: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], loss_out_0: pl.Tensor[[1], pl.FP32]) -> tuple[pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[25600, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 5120], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[5120, 25600], pl.FP32], pl.Tensor[[25600, 5120], pl.FP32], pl.Tensor[[64, 4096, 5120], pl.BFLOAT16], pl.Tensor[[1], pl.FP32]]:
        with pl.auto_incore():
            grad_wq_1: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wq_0, 0.0)
            grad_wk_1: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wk_0, 0.0)
            grad_wv_1: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wv_0, 0.0)
            grad_wo_1: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.mul(grad_wo_0, 0.0)
            grad_w_gate_1: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.mul(grad_w_gate_0, 0.0)
            grad_w_up_1: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.mul(grad_w_up_0, 0.0)
            grad_w_down_1: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.mul(grad_w_down_0, 0.0)
            loss_acc_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            loss_acc_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(loss_acc_0, 0.0)
            tok_blocks_0: pl.Scalar[pl.INDEX] = 4096 // 2
            for b_0, (loss_acc_iter_2, out_iter_1) in pl.parallel(0, 64, 1, init_values=(loss_acc_1, out_0), chunk=4):
                for p0_idx_0, (loss_acc_iter_4, out_iter_3) in pl.range(0, tok_blocks_0, 1, init_values=(loss_acc_iter_2, out_iter_1)):
                    p0_0: pl.Scalar[pl.INDEX] = p0_idx_0 * 2
                    sq_sum_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    sq_sum_1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
                    for kb_0, (sq_sum_iter_2,) in pl.range(0, 80, 1, init_values=(sq_sum_1,)):
                        k0_0: pl.Scalar[pl.INDEX] = kb_0 * 64
                        x_chunk_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states_0, [2, 64], [b_0, p0_0, k0_0]), target_type=pl.FP32, mode=2)
                        sq_sum_4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, pl.tensor.row_sum(pl.tensor.mul(x_chunk_0, x_chunk_0)))
                        sq_sum_3: pl.Tensor[[2, 1], pl.FP32] = pl.yield_(sq_sum_4)
                    inv_rms_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum_3, 0.000195313), 1e-06))
                    normed_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for kb_1, (k0_iter_1, normed_tile_iter_1, x_chunk_iter_1) in pl.range(0, 80, 1, init_values=(k0_0, normed_tile_0, x_chunk_0)):
                        k0_3: pl.Scalar[pl.INDEX] = kb_1 * 64
                        x_chunk_3: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states_0, [2, 64], [b_0, p0_0, k0_3]), target_type=pl.FP32, mode=2)
                        gamma_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 64], [0, k0_3])
                        normed_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_3, inv_rms_0), gamma_0)
                        normed_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(normed_tile_iter_1, pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2), [0, k0_3])
                        k0_2, normed_tile_2, x_chunk_2 = pl.yield_(k0_3, normed_tile_3, x_chunk_3)
                    q_proj_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    k_proj_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    v_proj_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for ob_0, (k0_iter_4, k_proj_tile_iter_1, kb_iter_2, q_proj_tile_iter_1, v_proj_tile_iter_1) in pl.range(0, 40, 1, init_values=(k0_2, k_proj_tile_0, kb_1, q_proj_tile_0, v_proj_tile_0)):
                        q0_0: pl.Scalar[pl.INDEX] = ob_0 * 128
                        q_acc_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        q_acc_1: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(q_acc_0, 0.0)
                        k_acc_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        k_acc_1: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(k_acc_0, 0.0)
                        v_acc_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        v_acc_1: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(v_acc_0, 0.0)
                        for kb_4, (k0_iter_6, k_acc_iter_2, q_acc_iter_2, v_acc_iter_2) in pl.range(0, 80, 1, init_values=(k0_iter_4, k_acc_1, q_acc_1, v_acc_1)):
                            k0_8: pl.Scalar[pl.INDEX] = kb_4 * 64
                            n_chunk_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(normed_tile_2, [2, 64], [0, k0_8]), target_type=pl.BFLOAT16, mode=2)
                            wq_c_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wq_0, [64, 128], [k0_8, q0_0])
                            wk_c_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wk_0, [64, 128], [k0_8, q0_0])
                            wv_c_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wv_0, [64, 128], [k0_8, q0_0])
                            q_acc_4: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(q_acc_iter_2, pl.tensor.matmul(n_chunk_0, wq_c_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                            k_acc_4: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(k_acc_iter_2, pl.tensor.matmul(n_chunk_0, wk_c_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                            v_acc_4: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(v_acc_iter_2, pl.tensor.matmul(n_chunk_0, wv_c_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                            k0_7, k_acc_3, q_acc_3, v_acc_3 = pl.yield_(k0_8, k_acc_4, q_acc_4, v_acc_4)
                        q_proj_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj_tile_iter_1, pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2), [0, q0_0])
                        k_proj_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(k_proj_tile_iter_1, pl.tensor.cast(k_acc_3, target_type=pl.BFLOAT16, mode=2), [0, q0_0])
                        v_proj_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(v_proj_tile_iter_1, pl.tensor.cast(v_acc_3, target_type=pl.BFLOAT16, mode=2), [0, q0_0])
                        k0_5, k_proj_tile_2, kb_3, q_proj_tile_2, v_proj_tile_2 = pl.yield_(k0_7, k_proj_tile_3, kb_4, q_proj_tile_3, v_proj_tile_3)
                    scores_0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.create([2, 2], dtype=pl.FP32)
                    scores_1: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.mul(scores_0, 0.0)
                    for kb_5, (k0_iter_9, scores_iter_2) in pl.range(0, 80, 1, init_values=(k0_5, scores_1)):
                        k0_11: pl.Scalar[pl.INDEX] = kb_5 * 64
                        q_c_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile_2, [2, 64], [0, k0_11]), target_type=pl.FP32, mode=2)
                        k_c_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(k_proj_tile_2, [2, 64], [0, k0_11]), target_type=pl.FP32, mode=2)
                        scores_4: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.add(scores_iter_2, pl.tensor.matmul(q_c_0, k_c_0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                        k0_10, scores_3 = pl.yield_(k0_11, scores_4)
                    scores_5: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.mul(scores_3, 0.0139754)
                    scores_exp_0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.exp(scores_5)
                    scores_sum_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(scores_exp_0)
                    attn_w_0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.row_expand_mul(scores_exp_0, pl.tensor.recip(scores_sum_0))
                    context_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for ob_1, (context_tile_iter_1,) in pl.range(0, 40, 1, init_values=(context_tile_0,)):
                        o0_0: pl.Scalar[pl.INDEX] = ob_1 * 128
                        v_c_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(v_proj_tile_2, [2, 128], [0, o0_0]), target_type=pl.FP32, mode=2)
                        ctx_c_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.matmul(attn_w_0, v_c_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        context_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(context_tile_iter_1, pl.tensor.cast(ctx_c_0, target_type=pl.BFLOAT16, mode=2), [0, o0_0])
                        context_tile_2: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.yield_(context_tile_3)
                    resid1_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    for ob_2, (ctx_c_iter_1, k0_iter_12, kb_iter_6, o0_iter_1, resid1_tile_iter_1) in pl.range(0, 40, 1, init_values=(ctx_c_0, k0_10, kb_5, o0_0, resid1_tile_0)):
                        o0_3: pl.Scalar[pl.INDEX] = ob_2 * 128
                        o_acc_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.create([2, 128], dtype=pl.FP32)
                        o_acc_1: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
                        for kb_8, (ctx_c_iter_3, k0_iter_14, o_acc_iter_2) in pl.range(0, 80, 1, init_values=(ctx_c_iter_1, k0_iter_12, o_acc_1)):
                            k0_16: pl.Scalar[pl.INDEX] = kb_8 * 64
                            ctx_c_5: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(context_tile_2, [2, 64], [0, k0_16])
                            wo_c_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(wo_0, [64, 128], [k0_16, o0_3])
                            o_acc_4: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(o_acc_iter_2, pl.tensor.matmul(ctx_c_5, wo_c_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            ctx_c_4, k0_15, o_acc_3 = pl.yield_(ctx_c_5, k0_16, o_acc_4)
                        resid_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states_0, [2, 128], [b_0, p0_0, o0_3]), target_type=pl.FP32, mode=2)
                        resid1_tile_3: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1, pl.tensor.add(o_acc_3, resid_0), [0, o0_3])
                        ctx_c_2, k0_13, kb_7, o0_2, resid1_tile_2 = pl.yield_(ctx_c_4, k0_15, kb_8, o0_3, resid1_tile_3)
                    sq_sum2_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    sq_sum2_1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(sq_sum2_0, 0.0)
                    for kb_9, (k0_iter_17, sq_sum2_iter_2, x_chunk_iter_4) in pl.range(0, 80, 1, init_values=(k0_13, sq_sum2_1, x_chunk_2)):
                        k0_19: pl.Scalar[pl.INDEX] = kb_9 * 64
                        x_chunk_6: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(resid1_tile_2, [2, 64], [0, k0_19])
                        sq_sum2_4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum2_iter_2, pl.tensor.row_sum(pl.tensor.mul(x_chunk_6, x_chunk_6)))
                        k0_18, sq_sum2_3, x_chunk_5 = pl.yield_(k0_19, sq_sum2_4, x_chunk_6)
                    inv_rms2_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum2_3, 0.000195313), 1e-06))
                    post_norm_tile_0: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.create([2, 5120], dtype=pl.BFLOAT16)
                    for kb_10, (gamma_iter_1, k0_iter_20, normed_iter_1, post_norm_tile_iter_1, x_chunk_iter_7) in pl.range(0, 80, 1, init_values=(gamma_0, k0_18, normed_0, post_norm_tile_0, x_chunk_5)):
                        k0_22: pl.Scalar[pl.INDEX] = kb_10 * 64
                        x_chunk_9: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(resid1_tile_2, [2, 64], [0, k0_22])
                        gamma_3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.view(post_rms_weight_0, [1, 64], [0, k0_22])
                        normed_3: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_9, inv_rms2_0), gamma_3)
                        post_norm_tile_3: pl.Tensor[[2, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile_iter_1, pl.tensor.cast(normed_3, target_type=pl.BFLOAT16, mode=2), [0, k0_22])
                        gamma_2, k0_21, normed_2, post_norm_tile_2, x_chunk_8 = pl.yield_(gamma_3, k0_22, normed_3, post_norm_tile_3, x_chunk_9)
                    down_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    down_tile_1: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(down_tile_0, 0.0)
                    for mb_0, (down_tile_iter_2, k0_iter_23, kb_iter_11, o0_iter_4, ob_iter_3) in pl.range(0, 100, 1, init_values=(down_tile_1, k0_21, kb_10, o0_2, ob_2)):
                        m0_0: pl.Scalar[pl.INDEX] = mb_0 * 256
                        gate_acc_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        up_acc_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        gate_acc_1: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(gate_acc_0, 0.0)
                        up_acc_1: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(up_acc_0, 0.0)
                        for kb_13, (gate_acc_iter_2, k0_iter_25, up_acc_iter_2) in pl.range(0, 80, 1, init_values=(gate_acc_1, k0_iter_23, up_acc_1)):
                            k0_27: pl.Scalar[pl.INDEX] = kb_13 * 64
                            post_chunk_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(post_norm_tile_2, [2, 64], [0, k0_27])
                            wg_0: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [64, 256], [k0_27, m0_0])
                            wu_0: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up_0, [64, 256], [k0_27, m0_0])
                            gate_acc_4: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(gate_acc_iter_2, pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_acc_4: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(up_acc_iter_2, pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            gate_acc_3, k0_26, up_acc_3 = pl.yield_(gate_acc_4, k0_27, up_acc_4)
                        sigmoid_chunk_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_acc_3)), 1.0))
                        mlp_chunk_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.mul(pl.tensor.mul(gate_acc_3, sigmoid_chunk_0), up_acc_3), target_type=pl.BFLOAT16, mode=2)
                        for ob_5, (down_tile_iter_4, o0_iter_6) in pl.range(0, 40, 1, init_values=(down_tile_iter_2, o0_iter_4)):
                            o0_8: pl.Scalar[pl.INDEX] = ob_5 * 128
                            down_prev_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.view(down_tile_iter_4, [2, 128], [0, o0_8])
                            wd_0: pl.Tensor[[256, 128], pl.BFLOAT16] = pl.tensor.view(w_down_0, [256, 128], [m0_0, o0_8])
                            down_part_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(down_prev_0, pl.tensor.matmul(mlp_chunk_0, wd_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            down_tile_6: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(down_tile_iter_4, down_part_0, [0, o0_8])
                            down_tile_5, o0_7 = pl.yield_(down_tile_6, o0_8)
                        down_tile_3, k0_24, kb_12, o0_5, ob_4 = pl.yield_(down_tile_5, k0_26, kb_13, o0_7, ob_5)
                    out_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    for ob_6, (o0_iter_9, out_iter_5, out_tile_iter_1) in pl.range(0, 40, 1, init_values=(o0_5, out_iter_3, out_tile_0)):
                        o0_11: pl.Scalar[pl.INDEX] = ob_6 * 128
                        out_chunk_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.add(pl.tensor.view(down_tile_3, [2, 128], [0, o0_11]), pl.tensor.view(resid1_tile_2, [2, 128], [0, o0_11]))
                        out_tile_3: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(out_tile_iter_1, out_chunk_0, [0, o0_11])
                        out_7: pl.Tensor[[64, 4096, 5120], pl.BFLOAT16] = pl.tensor.assemble(out_iter_5, pl.tensor.cast(out_chunk_0, target_type=pl.BFLOAT16, mode=2), [b_0, p0_0, o0_11])
                        o0_10, out_6, out_tile_2 = pl.yield_(o0_11, out_7, out_tile_3)
                    tgt_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.cast(pl.tensor.view(target_states_0, [2, 5120], [b_0, p0_0, 0]), target_type=pl.FP32, mode=2)
                    diff_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.sub(out_tile_2, tgt_tile_0)
                    sq_tile_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(diff_tile_0, diff_tile_0)
                    sq_row_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(sq_tile_0)
                    for ti_0, (loss_acc_iter_6,) in pl.range(0, 2, 1, init_values=(loss_acc_iter_4,)):
                        cur_0: pl.Scalar[pl.FP32] = pl.tensor.read(loss_acc_iter_6, [0, 0])
                        addv_0: pl.Scalar[pl.FP32] = pl.tensor.read(sq_row_0, [ti_0, 0])
                        acc_t_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                        acc_t_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(acc_t_0, 0.0)
                        acc_t_2: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(acc_t_1, cur_0 + addv_0)
                        loss_acc_8: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.assemble(loss_acc_iter_6, acc_t_2, [0, 0])
                        loss_acc_7: pl.Tensor[[1, 1], pl.FP32] = pl.yield_(loss_acc_8)
                    d_out_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(diff_tile_0, 0.000390625)
                    d_down_0: pl.Tensor[[2, 5120], pl.FP32] = d_out_0
                    d_resid1_bwd_0: pl.Tensor[[2, 5120], pl.FP32] = d_out_0
                    d_post_norm_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.create([2, 5120], dtype=pl.FP32)
                    d_post_norm_1: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.mul(d_post_norm_0, 0.0)
                    for mb_1, (d_post_norm_iter_2, k0_iter_28, kb_iter_14, m0_iter_1, o0_iter_12, ob_iter_7) in pl.range(0, 100, 1, init_values=(d_post_norm_1, k0_24, kb_12, m0_0, o0_10, ob_6)):
                        m0_3: pl.Scalar[pl.INDEX] = mb_1 * 256
                        gate_r_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        up_r_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        gate_r_1: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(gate_r_0, 0.0)
                        up_r_1: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(up_r_0, 0.0)
                        for kb_16, (gate_r_iter_2, k0_iter_30, up_r_iter_2) in pl.range(0, 80, 1, init_values=(gate_r_1, k0_iter_28, up_r_1)):
                            k0_32: pl.Scalar[pl.INDEX] = kb_16 * 64
                            post_c_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(post_norm_tile_2, [2, 64], [0, k0_32])
                            wg_c_0: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [64, 256], [k0_32, m0_3])
                            wu_c_0: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up_0, [64, 256], [k0_32, m0_3])
                            gate_r_4: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(gate_r_iter_2, pl.tensor.matmul(post_c_0, wg_c_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_r_4: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(up_r_iter_2, pl.tensor.matmul(post_c_0, wu_c_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            gate_r_3, k0_31, up_r_3 = pl.yield_(gate_r_4, k0_32, up_r_4)
                        sig_r_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_r_3)), 1.0))
                        d_mlp_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.create([2, 256], dtype=pl.FP32)
                        d_mlp_1: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(d_mlp_0, 0.0)
                        for ob_9, (d_mlp_iter_2, o0_iter_14) in pl.range(0, 40, 1, init_values=(d_mlp_1, o0_iter_12)):
                            o0_16: pl.Scalar[pl.INDEX] = ob_9 * 128
                            dd_c_0: pl.Tensor[[2, 128], pl.FP32] = pl.tensor.view(d_down_0, [2, 128], [0, o0_16])
                            wd_c_0: pl.Tensor[[256, 128], pl.BFLOAT16] = pl.tensor.view(w_down_0, [256, 128], [m0_3, o0_16])
                            d_mlp_4: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(d_mlp_iter_2, pl.tensor.matmul(dd_c_0, wd_c_0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                            d_mlp_3, o0_15 = pl.yield_(d_mlp_4, o0_16)
                        one_m_sig_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(sig_r_0, -1.0), 1.0)
                        silu_deriv_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(sig_r_0, pl.tensor.add(pl.tensor.mul(gate_r_3, one_m_sig_0), 1.0))
                        d_gate_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(pl.tensor.mul(d_mlp_3, up_r_3), silu_deriv_0)
                        d_up_0: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.mul(d_mlp_3, pl.tensor.mul(gate_r_3, sig_r_0))
                        for kb_17, (d_post_norm_iter_4, k0_iter_33, wg_c_iter_1, wu_c_iter_1) in pl.range(0, 80, 1, init_values=(d_post_norm_iter_2, k0_31, wg_c_0, wu_c_0)):
                            k0_35: pl.Scalar[pl.INDEX] = kb_17 * 64
                            dpn_old_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(d_post_norm_iter_4, [2, 64], [0, k0_35])
                            wg_c_3: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [64, 256], [k0_35, m0_3])
                            wu_c_3: pl.Tensor[[64, 256], pl.BFLOAT16] = pl.tensor.view(w_up_0, [64, 256], [k0_35, m0_3])
                            dpn_new_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(dpn_old_0, pl.tensor.add(pl.tensor.matmul(d_gate_0, wg_c_3, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32), pl.tensor.matmul(d_up_0, wu_c_3, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)))
                            d_post_norm_6: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.assemble(d_post_norm_iter_4, dpn_new_0, [0, k0_35])
                            d_post_norm_5, k0_34, wg_c_2, wu_c_2 = pl.yield_(d_post_norm_6, k0_35, wg_c_3, wu_c_3)
                        d_post_norm_3, k0_29, kb_15, m0_2, o0_13, ob_8 = pl.yield_(d_post_norm_5, k0_34, kb_17, m0_3, o0_15, ob_9)
                    d_resid1_0: pl.Tensor[[2, 5120], pl.FP32] = pl.tensor.add(d_resid1_bwd_0, d_post_norm_3)
                    bwd_energy_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32)
                    bwd_energy_1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(bwd_energy_0, 0.0)
                    for kb_18, (bwd_energy_iter_2, k0_iter_36, k_c_iter_1, q_c_iter_1, v_c_iter_1) in pl.range(0, 80, 1, init_values=(bwd_energy_1, k0_29, k_c_0, q_c_0, v_c_0)):
                        k0_38: pl.Scalar[pl.INDEX] = kb_18 * 64
                        dr_c_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(d_resid1_0, [2, 64], [0, k0_38])
                        q_c_3: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj_tile_2, [2, 64], [0, k0_38]), target_type=pl.FP32, mode=2)
                        k_c_3: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(k_proj_tile_2, [2, 64], [0, k0_38]), target_type=pl.FP32, mode=2)
                        v_c_3: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(v_proj_tile_2, [2, 64], [0, k0_38]), target_type=pl.FP32, mode=2)
                        contrib_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(dr_c_0, pl.tensor.add(pl.tensor.add(q_c_3, k_c_3), v_c_3)))
                        bwd_energy_4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy_iter_2, contrib_0)
                        bwd_energy_3, k0_37, k_c_2, q_c_2, v_c_2 = pl.yield_(bwd_energy_4, k0_38, k_c_3, q_c_3, v_c_3)
                    bwd_energy_5: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy_3, pl.tensor.row_sum(attn_w_0))
                    grad_sink_0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.mul(bwd_energy_5, 0.0)
                    loss_acc_5, out_4 = pl.yield_(loss_acc_7, out_6)
                loss_acc_3, out_2 = pl.yield_(loss_acc_5, out_4)
            proxy_mlp_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_up_0, [2, 256], [0, 0]), target_type=pl.BFLOAT16, mode=2)
            for qb_0, (grad_w_down_iter_2, mom_w_down_iter_1, q0_iter_1) in pl.range(0, 40, 1, init_values=(grad_w_down_1, mom_w_down_0, q0_0)):
                q0_3: pl.Scalar[pl.INDEX] = qb_0 * 128
                proxy_go_0: pl.Tensor[[2, 128], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(target_states_0, [2, 128], [0, 0, q0_3]), target_type=pl.BFLOAT16, mode=2)
                grad_down_raw_0: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.matmul(proxy_mlp_0, proxy_go_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_down_prev_0: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.view(mom_w_down_iter_1, [256, 128], [0, q0_3])
                mom_down_new_0: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_down_prev_0, 0.95), pl.tensor.mul(grad_down_raw_0, 0.05))
                muon_down_0: pl.Tensor[[256, 128], pl.FP32] = mom_down_new_0
                for __0, (muon_down_iter_1,) in pl.range(0, 2, 1, init_values=(muon_down_0,)):
                    gram_0: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_down_iter_1, muon_down_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_down_3: pl.Tensor[[256, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_down_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_down_iter_1, gram_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    muon_down_2: pl.Tensor[[256, 128], pl.FP32] = pl.yield_(muon_down_3)
                grad_w_down_4: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.assemble(grad_w_down_iter_2, pl.tensor.mul(muon_down_2, -0.0002), [0, q0_3])
                mom_w_down_3: pl.Tensor[[25600, 5120], pl.FP32] = pl.tensor.assemble(mom_w_down_iter_1, mom_down_new_0, [0, q0_3])
                grad_w_down_3, mom_w_down_2, q0_2 = pl.yield_(grad_w_down_4, mom_w_down_3, q0_3)
            proxy_ctx_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(wq_0, [2, 64], [0, 0])
            proxy_n_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(hidden_states_0, [2, 64], [0, 0, 0]), target_type=pl.BFLOAT16, mode=2)
            for qb_1, (__iter_1, grad_wk_iter_2, grad_wo_iter_2, grad_wq_iter_2, grad_wv_iter_2, gram_iter_1, mom_wk_iter_1, mom_wo_iter_1, mom_wq_iter_1, mom_wv_iter_1, q0_iter_4) in pl.range(0, 40, 1, init_values=(__0, grad_wk_1, grad_wo_1, grad_wq_1, grad_wv_1, gram_0, mom_wk_0, mom_wo_0, mom_wq_0, mom_wv_0, q0_2)):
                q0_6: pl.Scalar[pl.INDEX] = qb_1 * 128
                proxy_tgt_0: pl.Tensor[[2, 128], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(target_states_0, [2, 128], [0, 0, q0_6]), target_type=pl.BFLOAT16, mode=2)
                grad_wo_raw_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_ctx_0, proxy_tgt_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wo_prev_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wo_iter_1, [64, 128], [0, q0_6])
                mom_wo_new_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wo_prev_0, 0.95), pl.tensor.mul(grad_wo_raw_0, 0.05))
                muon_wo_0: pl.Tensor[[64, 128], pl.FP32] = mom_wo_new_0
                for __3, (gram_iter_3, muon_wo_iter_1) in pl.range(0, 2, 1, init_values=(gram_iter_1, muon_wo_0)):
                    gram_5: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wo_iter_1, muon_wo_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wo_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wo_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wo_iter_1, gram_5, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    gram_4, muon_wo_2 = pl.yield_(gram_5, muon_wo_3)
                grad_wo_4: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wo_iter_2, pl.tensor.mul(muon_wo_2, -0.0002), [0, q0_6])
                mom_wo_3: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wo_iter_1, mom_wo_new_0, [0, q0_6])
                grad_wq_raw_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n_0, proxy_tgt_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wq_prev_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wq_iter_1, [64, 128], [0, q0_6])
                mom_wq_new_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wq_prev_0, 0.95), pl.tensor.mul(grad_wq_raw_0, 0.05))
                muon_wq_0: pl.Tensor[[64, 128], pl.FP32] = mom_wq_new_0
                for __4, (gram_iter_6, muon_wq_iter_1) in pl.range(0, 2, 1, init_values=(gram_4, muon_wq_0)):
                    gram_8: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wq_iter_1, muon_wq_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wq_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wq_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wq_iter_1, gram_8, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    gram_7, muon_wq_2 = pl.yield_(gram_8, muon_wq_3)
                grad_wq_4: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wq_iter_2, pl.tensor.mul(muon_wq_2, -0.0002), [0, q0_6])
                mom_wq_3: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wq_iter_1, mom_wq_new_0, [0, q0_6])
                grad_wk_raw_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n_0, proxy_tgt_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wk_prev_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wk_iter_1, [64, 128], [0, q0_6])
                mom_wk_new_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wk_prev_0, 0.95), pl.tensor.mul(grad_wk_raw_0, 0.05))
                muon_wk_0: pl.Tensor[[64, 128], pl.FP32] = mom_wk_new_0
                for __5, (gram_iter_9, muon_wk_iter_1) in pl.range(0, 2, 1, init_values=(gram_7, muon_wk_0)):
                    gram_11: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wk_iter_1, muon_wk_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wk_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wk_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wk_iter_1, gram_11, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    gram_10, muon_wk_2 = pl.yield_(gram_11, muon_wk_3)
                grad_wk_4: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wk_iter_2, pl.tensor.mul(muon_wk_2, -0.0002), [0, q0_6])
                mom_wk_3: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wk_iter_1, mom_wk_new_0, [0, q0_6])
                grad_wv_raw_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(proxy_n_0, proxy_tgt_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wv_prev_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.view(mom_wv_iter_1, [64, 128], [0, q0_6])
                mom_wv_new_0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wv_prev_0, 0.95), pl.tensor.mul(grad_wv_raw_0, 0.05))
                muon_wv_0: pl.Tensor[[64, 128], pl.FP32] = mom_wv_new_0
                for __6, (gram_iter_12, muon_wv_iter_1) in pl.range(0, 2, 1, init_values=(gram_10, muon_wv_0)):
                    gram_14: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(muon_wv_iter_1, muon_wv_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wv_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wv_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wv_iter_1, gram_14, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    gram_13, muon_wv_2 = pl.yield_(gram_14, muon_wv_3)
                grad_wv_4: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(grad_wv_iter_2, pl.tensor.mul(muon_wv_2, -0.0002), [0, q0_6])
                mom_wv_3: pl.Tensor[[5120, 5120], pl.FP32] = pl.tensor.assemble(mom_wv_iter_1, mom_wv_new_0, [0, q0_6])
                __2, grad_wk_3, grad_wo_3, grad_wq_3, grad_wv_3, gram_2, mom_wk_2, mom_wo_2, mom_wq_2, mom_wv_2, q0_5 = pl.yield_(__6, grad_wk_4, grad_wo_4, grad_wq_4, grad_wv_4, gram_13, mom_wk_3, mom_wo_3, mom_wq_3, mom_wv_3, q0_6)
            proxy_post_0: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(hidden_states_0, [2, 64], [0, 0, 64]), target_type=pl.BFLOAT16, mode=2)
            for mb_2, (__iter_7, grad_w_gate_iter_2, grad_w_up_iter_2, m0_iter_4, mom_w_gate_iter_1, mom_w_up_iter_1) in pl.range(0, 100, 1, init_values=(__2, grad_w_gate_1, grad_w_up_1, m0_2, mom_w_gate_0, mom_w_up_0)):
                m0_6: pl.Scalar[pl.INDEX] = mb_2 * 256
                proxy_gg_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_gate_0, [2, 256], [0, m0_6]), target_type=pl.BFLOAT16, mode=2)
                proxy_gu_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(w_up_0, [2, 256], [0, m0_6]), target_type=pl.BFLOAT16, mode=2)
                grad_wg_raw_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.matmul(proxy_post_0, proxy_gg_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                grad_wu_raw_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.matmul(proxy_post_0, proxy_gu_0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wg_prev_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.view(mom_w_gate_iter_1, [64, 256], [0, m0_6])
                mom_wu_prev_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.view(mom_w_up_iter_1, [64, 256], [0, m0_6])
                mom_wg_new_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wg_prev_0, 0.95), pl.tensor.mul(grad_wg_raw_0, 0.05))
                mom_wu_new_0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(mom_wu_prev_0, 0.95), pl.tensor.mul(grad_wu_raw_0, 0.05))
                muon_wg_0: pl.Tensor[[64, 256], pl.FP32] = mom_wg_new_0
                muon_wu_0: pl.Tensor[[64, 256], pl.FP32] = mom_wu_new_0
                for __9, (muon_wg_iter_1, muon_wu_iter_1) in pl.range(0, 2, 1, init_values=(muon_wg_0, muon_wu_0)):
                    gram_wg_0: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(muon_wg_iter_1, muon_wg_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram_wu_0: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(muon_wu_iter_1, muon_wu_iter_1, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    muon_wg_3: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wg_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wg_iter_1, gram_wg_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    muon_wu_3: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.add(pl.tensor.mul(muon_wu_iter_1, 1.5), pl.tensor.mul(pl.tensor.matmul(muon_wu_iter_1, gram_wu_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                    muon_wg_2, muon_wu_2 = pl.yield_(muon_wg_3, muon_wu_3)
                grad_w_gate_4: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(grad_w_gate_iter_2, pl.tensor.mul(muon_wg_2, -0.0002), [0, m0_6])
                grad_w_up_4: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(grad_w_up_iter_2, pl.tensor.mul(muon_wu_2, -0.0002), [0, m0_6])
                mom_w_gate_3: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(mom_w_gate_iter_1, mom_wg_new_0, [0, m0_6])
                mom_w_up_3: pl.Tensor[[5120, 25600], pl.FP32] = pl.tensor.assemble(mom_w_up_iter_1, mom_wu_new_0, [0, m0_6])
                __8, grad_w_gate_3, grad_w_up_3, m0_5, mom_w_gate_2, mom_w_up_2 = pl.yield_(__9, grad_w_gate_4, grad_w_up_4, m0_6, mom_w_gate_3, mom_w_up_3)
            loss_vec_0: pl.Tensor[[1], pl.FP32] = pl.tensor.view(loss_acc_3, [1], [0, 0])
            loss_out_1: pl.Tensor[[1], pl.FP32] = pl.tensor.assemble(loss_out_0, loss_vec_0, [0])
        return grad_wq_3, grad_wk_3, grad_wv_3, grad_wo_3, grad_w_gate_3, grad_w_up_3, grad_w_down_3, mom_wq_2, mom_wk_2, mom_wv_2, mom_wo_2, mom_w_gate_2, mom_w_up_2, mom_w_down_2, out_2, loss_out_1