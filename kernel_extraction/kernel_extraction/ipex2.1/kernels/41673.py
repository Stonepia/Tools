

# Original file: ./AllenaiLongformerBase__0_forward_61.0/AllenaiLongformerBase__0_forward_61.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ca/ccanskdxuf6af4rgmgutlr33wirikebrz2abjygzgkoaqiwxj2nf.py
# Source Nodes: [add, add_1, add_2, int_1, l__self___embeddings_dropout, l__self___embeddings_layer_norm, l__self___embeddings_position_embeddings, l__self___embeddings_token_type_embeddings, l__self___embeddings_word_embeddings, long, mul_1, ne, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_dropout, aten.native_layer_norm, aten.ne]
# add => add
# add_1 => add_1
# add_2 => add_2
# int_1 => convert_element_type_1
# l__self___embeddings_dropout => gt, mul_4, mul_5
# l__self___embeddings_layer_norm => add_3, add_4, convert_element_type_4, convert_element_type_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# l__self___embeddings_position_embeddings => embedding_1
# l__self___embeddings_token_type_embeddings => embedding_2
# l__self___embeddings_word_embeddings => embedding
# long => convert_element_type_3
# mul_1 => mul_1
# ne => ne
# type_as => convert_element_type_2
triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_3 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_3', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*i64', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*fp32', 11: '*i1', 12: '*bf16', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_native_dropout_native_layer_norm_ne_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
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
    tmp14 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.int32)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7 + tmp3
    tmp9 = tl.where(tmp2 < 0, tmp2 + 50265, tmp2)
    # tl.device_assert((0 <= tmp9) & (tmp9 < 50265), "index out of bounds: 0 <= tmp9 < 50265")
    tmp10 = tl.load(in_ptr1 + (r1 + (768*tmp9)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp8 < 0, tmp8 + 4098, tmp8)
    # tl.device_assert((0 <= tmp11) & (tmp11 < 4098), "index out of bounds: 0 <= tmp11 < 4098")
    tmp12 = tl.load(in_ptr2 + (r1 + (768*tmp11)), rmask, other=0.0).to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 768, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = 768.0
    tmp34 = tmp32 / tmp33
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tl.load(in_ptr4 + load_seed_offset)
    tmp39 = r1 + (768*x0)
    tmp40 = tl.rand(tmp38, (tmp39).to(tl.uint32))
    tmp41 = tmp40.to(tl.float32)
    tmp42 = 0.1
    tmp43 = tmp41 > tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp16 - tmp26
    tmp46 = tmp45 * tmp37
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp44 * tmp53
    tmp55 = 1.1111111111111112
    tmp56 = tmp54 * tmp55
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp15, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp37, None)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp43, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp56, rmask)
    tl.store(out_ptr1 + (x0), tmp26, None)
''')
