

# Original file: ./hf_T5_generate__44_inference_84.24/hf_T5_generate__44_inference_84.24_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/yg/cygudj2bryyxttbr63eg4xe5ur36m5ncormunnqb4ulszssbnzv3.py
# Source Nodes: [add_50, add_52, add_53, any_18, clamp_15, clamp_16, clamp_17, isinf_17, l__self___decoder_block_5_layer_1_dropout, l__self___decoder_block_5_layer__1__dropout, mean_18, mul_40, mul_41, mul_42, neg_16, neg_17, neg_18, pow_19, rsqrt_18, to_40, to_41, where_1, where_16, where_17, where_18], Original ATen: [aten._to_copy, aten.add, aten.any, aten.clamp, aten.clone, aten.isinf, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_50 => add_54
# add_52 => add_56
# add_53 => add_57
# any_18 => any_18
# clamp_15 => clamp_max_15, clamp_min_15, convert_element_type_89, convert_element_type_90
# clamp_16 => clamp_max_16, clamp_min_16, convert_element_type_95, convert_element_type_96
# clamp_17 => clamp_max_17, clamp_min_17, convert_element_type_100, convert_element_type_99
# isinf_17 => isinf_17
# l__self___decoder_block_5_layer_1_dropout => clone_34
# l__self___decoder_block_5_layer__1__dropout => clone_36
# mean_18 => mean_18
# mul_40 => mul_40
# mul_41 => mul_41
# mul_42 => mul_42
# neg_16 => neg_16
# neg_17 => neg_17
# neg_18 => neg_18
# pow_19 => pow_19
# rsqrt_18 => rsqrt_18
# to_40 => convert_element_type_101
# to_41 => convert_element_type_102
# where_1 => full_default_1, full_default_2
# where_16 => where_16
# where_17 => where_17
# where_18 => where_18
triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_14 = async_compile.triton('triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_14', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (0))
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
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
    tmp49 = 0.04419417382415922
    tmp50 = tmp48 * tmp49
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp50, rmask)
''')
