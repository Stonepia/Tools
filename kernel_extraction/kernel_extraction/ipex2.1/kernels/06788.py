

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/4p/c4pylj7cw4tljer2unnlpsp3ciqqzabimklxu3mnfo2kcbvalbsi.py
# Source Nodes: [add, l__mod___distilbert_embeddings_dropout, l__mod___distilbert_embeddings_layer_norm, l__mod___distilbert_embeddings_word_embeddings], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm]
# add => add
# l__mod___distilbert_embeddings_dropout => gt, mul_2, mul_3
# l__mod___distilbert_embeddings_layer_norm => add_1, add_2, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# l__mod___distilbert_embeddings_word_embeddings => embedding
triton_per_fused_add_embedding_native_dropout_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*i64', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*i1', 10: '*fp16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 30522, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 30522), "index out of bounds: 0 <= tmp1 < 30522")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.full([1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp22 = 768.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tl.load(in_ptr3 + load_seed_offset)
    tmp28 = r1 + (768*x0)
    tmp29 = tl.rand(tmp27, (tmp28).to(tl.uint32))
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 0.1
    tmp32 = tmp30 > tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp5 - tmp15
    tmp35 = tmp34 * tmp26
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 * tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 + tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp33 * tmp42
    tmp44 = 1.1111111111111112
    tmp45 = tmp43 * tmp44
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, None)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp32, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp45, rmask)
    tl.store(out_ptr1 + (x0), tmp15, None)
''')
