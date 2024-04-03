

# Original file: ./hf_T5_generate__22_inference_62.2/hf_T5_generate__22_inference_62.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/fv/cfv5y37ynmpdhlbgwg2t62hhzmmvtzgvaybosv35uj6gdk464zge.py
# Source Nodes: [add_4, add_5, any_1, clamp, isinf, l__self___decoder_block_0_layer_0_dropout, l__self___decoder_embed_tokens, mean_1, mul_6, mul_7, neg_1, pow_2, rsqrt_1, to_6, to_7, where_1], Original ATen: [aten._to_copy, aten.add, aten.any, aten.clamp, aten.clone, aten.embedding, aten.isinf, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_4 => add_5
# add_5 => add_6
# any_1 => any_1
# clamp => clamp_max, clamp_min, convert_element_type_10, convert_element_type_9
# isinf => isinf
# l__self___decoder_block_0_layer_0_dropout => clone_2
# l__self___decoder_embed_tokens => embedding
# mean_1 => mean_1
# mul_6 => mul_6
# mul_7 => mul_7
# neg_1 => neg_1
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# to_6 => convert_element_type_11
# to_7 => convert_element_type_12
# where_1 => full_default_2, full_default_3, where_1
triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_3 = async_compile.triton('triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_3', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0).to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.isinf(tmp5).to(tl.int1)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp9, 0))
    tmp11 = tmp5.to(tl.float32)
    tmp12 = 64504.0
    tmp13 = 65504.0
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = -tmp14
    tmp16 = triton_helpers.maximum(tmp11, tmp15)
    tmp17 = triton_helpers.minimum(tmp16, tmp14)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = 512.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp19 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp25 * tmp32
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp33, rmask)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp10, None)
''')
