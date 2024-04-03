

# Original file: ./AllenaiLongformerBase__0_forward_61.0/AllenaiLongformerBase__0_forward_61.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/s4/cs4vvskzldvy4xkjp4kbs3nneetejgxflob4wth66e5q5benpzmh.py
# Source Nodes: [add, add_1, add_2, int_1, l__self___embeddings_dropout, l__self___embeddings_layer_norm, l__self___embeddings_position_embeddings, l__self___embeddings_token_type_embeddings, l__self___embeddings_word_embeddings, long, mul_1, ne, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.ne]
# add => add
# add_1 => add_1
# add_2 => add_2
# int_1 => convert_element_type
# l__self___embeddings_dropout => gt, mul_4, mul_5
# l__self___embeddings_layer_norm => add_3, add_4, mul_2, mul_3, rsqrt, sub_1, var_mean
# l__self___embeddings_position_embeddings => embedding_1
# l__self___embeddings_token_type_embeddings => embedding_2
# l__self___embeddings_word_embeddings => embedding
# long => convert_element_type_2
# mul_1 => mul_1
# ne => ne
# type_as => convert_element_type_1
triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_3 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_3', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_native_layer_norm_backward_ne_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr4, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7 + tmp3
    tmp9 = tl.where(tmp2 < 0, tmp2 + 50265, tmp2)
    # tl.device_assert((0 <= tmp9) & (tmp9 < 50265), "index out of bounds: 0 <= tmp9 < 50265")
    tmp10 = tl.load(in_ptr1 + (r1 + (768*tmp9)), rmask, other=0.0)
    tmp11 = tl.where(tmp8 < 0, tmp8 + 4098, tmp8)
    # tl.device_assert((0 <= tmp11) & (tmp11 < 4098), "index out of bounds: 0 <= tmp11 < 4098")
    tmp12 = tl.load(in_ptr2 + (r1 + (768*tmp11)), rmask, other=0.0)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tl.load(in_ptr4 + load_seed_offset)
    tmp33 = r1 + (768*x0)
    tmp34 = tl.rand(tmp32, (tmp33).to(tl.uint32))
    tmp35 = 0.1
    tmp36 = tmp34 > tmp35
    tmp37 = tmp15 - tmp25
    tmp38 = 768.0
    tmp39 = tmp31 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp44 = tmp36.to(tl.float32)
    tmp46 = tmp43 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp44 * tmp48
    tmp50 = 1.1111111111111112
    tmp51 = tmp49 * tmp50
    tmp52 = tmp42 / tmp38
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp36, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp43, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp51, rmask)
    tl.store(out_ptr7 + (x0), tmp52, None)
''')
