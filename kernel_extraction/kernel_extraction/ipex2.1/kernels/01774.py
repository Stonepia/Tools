

# Original file: ./hf_T5_generate__27_inference_67.7/hf_T5_generate__27_inference_67.7_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/qi/cqiv6bd3zhvnuh4icxxgarx4wfduqqymzywqsyrlzg5eex5oo4b4.py
# Source Nodes: [add_12, add_13, any_3, clamp_1, clamp_2, isinf_2, l__self___decoder_block_0_layer__1__dropout, mean_3, mul_10, mul_11, neg_2, neg_3, pow_4, rsqrt_3, to_10, to_11, where_1, where_2, where_3], Original ATen: [aten._to_copy, aten.add, aten.any, aten.clamp, aten.clone, aten.isinf, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_12 => add_16
# add_13 => add_17
# any_3 => any_3
# clamp_1 => clamp_max_1, clamp_min_1, convert_element_type_15, convert_element_type_16
# clamp_2 => clamp_max_2, clamp_min_2, convert_element_type_19, convert_element_type_20
# isinf_2 => isinf_2
# l__self___decoder_block_0_layer__1__dropout => clone_6
# mean_3 => mean_3
# mul_10 => mul_10
# mul_11 => mul_11
# neg_2 => neg_2
# neg_3 => neg_3
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# to_10 => convert_element_type_21
# to_11 => convert_element_type_22
# where_1 => full_default_1, full_default_2
# where_2 => where_2
# where_3 => where_3
triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_11 = async_compile.triton('triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_any_clamp_clone_isinf_mean_mul_neg_pow_rsqrt_scalar_tensor_where_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, xnumel, rnumel):
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
    tmp11 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = 64504.0
    tmp5 = 65504.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = -tmp6
    tmp8 = triton_helpers.maximum(tmp1, tmp7)
    tmp9 = triton_helpers.minimum(tmp8, tmp6)
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.isinf(tmp12).to(tl.int1)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp16, 0))
    tmp18 = tmp12.to(tl.float32)
    tmp19 = tl.where(tmp17, tmp4, tmp5)
    tmp20 = -tmp19
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp22 = triton_helpers.minimum(tmp21, tmp19)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = 512.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp24 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp30 * tmp37
    tl.store(out_ptr2 + (tl.broadcast_to(r0, [RBLOCK])), tmp38, rmask)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp17, None)
''')
