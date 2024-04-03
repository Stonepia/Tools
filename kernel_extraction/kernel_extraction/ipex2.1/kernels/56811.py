

# Original file: ./hf_T5_generate__22_inference_62.2/hf_T5_generate__22_inference_62.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/h4/ch4grmrmb6rvkdr4pj2emioei2t6y34ohbti25pdedhglwl5jpqg.py
# Source Nodes: [add_21, add_23, add_24, any_10, clamp_7, clamp_8, clamp_9, isinf_9, l__self___decoder_block_2_layer__1__dropout, l__self___decoder_block_3_layer_0_dropout, mean_10, mul_24, mul_25, neg_10, neg_8, neg_9, pow_11, rsqrt_10, to_24, to_25, where_1, where_10, where_8, where_9], Original ATen: [aten._to_copy, aten.add, aten.any, aten.clamp, aten.clone, aten.isinf, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_21 => add_27
# add_23 => add_30
# add_24 => add_31
# any_10 => any_10
# clamp_7 => clamp_max_7, clamp_min_7, convert_element_type_47, convert_element_type_48
# clamp_8 => clamp_max_8, clamp_min_8, convert_element_type_51, convert_element_type_52
# clamp_9 => clamp_max_9, clamp_min_9, convert_element_type_57, convert_element_type_58
# isinf_9 => isinf_9
# l__self___decoder_block_2_layer__1__dropout => clone_18
# l__self___decoder_block_3_layer_0_dropout => clone_20
# mean_10 => mean_10
# mul_24 => mul_24
# mul_25 => mul_25
# neg_10 => neg_10
# neg_8 => neg_8
# neg_9 => neg_9
# pow_11 => pow_11
# rsqrt_10 => rsqrt_10
# to_24 => convert_element_type_59
# to_25 => convert_element_type_60
# where_1 => full_default_2, full_default_3
# where_10 => where_10
# where_8 => where_8
# where_9 => where_9
triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_10 = async_compile.triton('triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_10', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*i1', 4: '*fp16', 5: '*fp16', 6: '*i1', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (0))
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp21 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = 64504.0
    tmp5 = 65504.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = -tmp6
    tmp8 = triton_helpers.maximum(tmp1, tmp7)
    tmp9 = triton_helpers.minimum(tmp8, tmp6)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tl.where(tmp15, tmp4, tmp5)
    tmp17 = -tmp16
    tmp18 = triton_helpers.maximum(tmp13, tmp17)
    tmp19 = triton_helpers.minimum(tmp18, tmp16)
    tmp20 = tmp19.to(tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.isinf(tmp22).to(tl.int1)
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp26, 0))
    tmp28 = tmp22.to(tl.float32)
    tmp29 = tl.where(tmp27, tmp4, tmp5)
    tmp30 = -tmp29
    tmp31 = triton_helpers.maximum(tmp28, tmp30)
    tmp32 = triton_helpers.minimum(tmp31, tmp29)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp41 = 512.0
    tmp42 = tmp39 / tmp41
    tmp43 = 1e-06
    tmp44 = tmp42 + tmp43
    tmp45 = libdevice.rsqrt(tmp44)
    tmp46 = tmp34 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp40 * tmp47
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp22, rmask)
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp48, rmask)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp27, None)
''')
