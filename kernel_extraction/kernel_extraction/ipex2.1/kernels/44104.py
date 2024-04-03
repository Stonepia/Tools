

# Original file: ./hf_T5_generate__35_inference_75.15/hf_T5_generate__35_inference_75.15_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/jb/cjbgv7dty4nssknpauc5s4po3jzuf34gr6tsyol6rpt7f7bcfygh.py
# Source Nodes: [add_10, add_11, add_6, any_2, clamp, clamp_1, isinf_1, l__self___decoder_block_0_layer_0_dropout, l__self___decoder_block_0_layer_1_dropout, l__self___decoder_embed_tokens, mean_2, mul_8, mul_9, neg_1, neg_2, pow_3, rsqrt_2, to_8, to_9, where_1, where_2], Original ATen: [aten._to_copy, aten.add, aten.any, aten.clamp, aten.clone, aten.embedding, aten.isinf, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_10 => add_14
# add_11 => add_15
# add_6 => add_9
# any_2 => any_2
# clamp => clamp_max, clamp_min, convert_element_type_10, convert_element_type_9
# clamp_1 => clamp_max_1, clamp_min_1, convert_element_type_15, convert_element_type_16
# isinf_1 => isinf_1
# l__self___decoder_block_0_layer_0_dropout => clone_2
# l__self___decoder_block_0_layer_1_dropout => clone_4
# l__self___decoder_embed_tokens => embedding
# mean_2 => mean_2
# mul_8 => mul_8
# mul_9 => mul_9
# neg_1 => neg_1
# neg_2 => neg_2
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# to_8 => convert_element_type_17
# to_9 => convert_element_type_18
# where_1 => full_default_1, full_default_2, where_1
# where_2 => where_2
triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_9 = async_compile.triton('triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_9', '''
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*fp16', 6: '*i1', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_any_clamp_clone_embedding_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel):
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
    tmp4 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp16 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0).to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp9 = 64504.0
    tmp10 = 65504.0
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = -tmp11
    tmp13 = triton_helpers.maximum(tmp6, tmp12)
    tmp14 = triton_helpers.minimum(tmp13, tmp11)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.isinf(tmp17).to(tl.int1)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp21, 0))
    tmp23 = tmp17.to(tl.float32)
    tmp24 = tl.where(tmp22, tmp9, tmp10)
    tmp25 = -tmp24
    tmp26 = triton_helpers.maximum(tmp23, tmp25)
    tmp27 = triton_helpers.minimum(tmp26, tmp24)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = 512.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp29 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp35 * tmp42
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp17, rmask)
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp43, rmask)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp22, None)
''')
