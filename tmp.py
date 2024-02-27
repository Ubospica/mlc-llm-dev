import tvm
from tvm.script import relax as R, tir as T, ir as I

@I.ir_module
class Module:
    @T.prim_func
    def apply_bitmask(
        var_logits: T.handle,
        var_seq_ids: T.handle,
        var_bitmask: T.handle,
    ) -> None:
        """Function that applies vocabulary masking in place."""
        T.func_attr(
            {"global_symbol": "apply_bitmask_inplace", "tir.noalias": True, "tir.is_scheduled": True}
        )
        batch_size = T.int32(is_size_var=True)
        vocab_size = T.int32(is_size_var=True)
        num_seq = T.int32(is_size_var=True)
        logits = T.match_buffer(var_logits, (batch_size, vocab_size), "float32")
        seq_ids = T.match_buffer(var_seq_ids, (num_seq,), "int32")
        bitmask = T.match_buffer(var_bitmask, (num_seq, (vocab_size + 31 // 32)), "int32")

        for fused_s_v_0 in T.thread_binding(0, (num_seq * vocab_size + 1023) // 1024, "blockIdx.x"):
            for fused_s_v_1 in T.thread_binding(0, 1024, "threadIdx.x"):
                with T.block("block"):
                    vs = T.axis.spatial(num_seq, (fused_s_v_0 * 1024 + fused_s_v_1) // vocab_size)
                    vv = T.axis.spatial(vocab_size, (fused_s_v_0 * 1024 + fused_s_v_1) % vocab_size)
                    T.where(fused_s_v_0 * 1024 + fused_s_v_1 < num_seq * vocab_size)
                    logits[seq_ids[vs], vv] = T.if_then_else(
                        (bitmask[vs, vv // 32] >> (vv % 32)) & 1 == 1,
                        logits[seq_ids[vs], vv],
                        T.float32(-1e10),
                    )


