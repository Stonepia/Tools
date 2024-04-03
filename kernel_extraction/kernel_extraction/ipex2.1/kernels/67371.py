

# Original file: ./hf_Longformer___60.0/hf_Longformer___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/pa/cpaaca24ewnxaprspwl3trfegbjmgg6m5syibwvshqepxgt675dr.py
# Source Nodes: [add, add_1, add_2, int_1, l__self___embeddings_layer_norm, l__self___embeddings_position_embeddings, l__self___embeddings_token_type_embeddings, l__self___embeddings_word_embeddings, long, mul_1, ne, type_as, zeros], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mul, aten.native_layer_norm, aten.ne, aten.zeros]
# add => add
# add_1 => add_1
# add_2 => add_2
# int_1 => convert_element_type_1
# l__self___embeddings_layer_norm => add_3, add_4, convert_element_type_4, convert_element_type_5, mul_2, mul_3, rsqrt, sub_1, var_mean
# l__self___embeddings_position_embeddings => embedding_1
# l__self___embeddings_token_type_embeddings => embedding_2
# l__self___embeddings_word_embeddings => embedding
# long => convert_element_type_3
# mul_1 => mul_1
# ne => ne
# type_as => convert_element_type_2
# zeros => full_default
triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mul_native_layer_norm_ne_zeros_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 50265, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 50265), "index out of bounds: 0 <= tmp1 < 50265")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.int32)
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp0 != tmp5
    tmp7 = tmp6.to(tl.int32)
    tmp8 = tmp4 * tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp9 + tmp5
    tmp11 = tl.where(tmp10 < 0, tmp10 + 4098, tmp10)
    # tl.device_assert((0 <= tmp11) & (tmp11 < 4098), "index out of bounds: 0 <= tmp11 < 4098")
    tmp12 = tl.load(in_ptr3 + (r1 + (768*tmp11)), rmask, other=0.0).to(tl.float32)
    tmp13 = tmp2 + tmp12
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
    tmp33 = tmp16 - tmp26
    tmp34 = 768.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp39 * tmp41
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 + tmp44
    tmp46 = tmp45.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp46, rmask)
''')
