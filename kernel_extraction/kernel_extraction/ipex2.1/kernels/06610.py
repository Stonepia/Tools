

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/yu/cyuswssmn667yhtdxjkhw2akd42zqhlerpfyvw4fribnq4o5qzwh.py
# Source Nodes: [add_128, clamp_59, mean_61, mul_129, mul_130, mul_131, neg_60, pow_62, rsqrt_61, to_129, to_130, where_1, where_61], Original ATen: [aten._to_copy, aten.add, aten.clamp, aten.mean, aten.mul, aten.neg, aten.pow, aten.rsqrt, aten.scalar_tensor, aten.where]
# add_128 => add_165
# clamp_59 => clamp_max_59, clamp_min_59, convert_element_type_321, convert_element_type_322
# mean_61 => mean_61
# mul_129 => mul_129
# mul_130 => mul_130
# mul_131 => mul_131
# neg_60 => neg_60
# pow_62 => pow_62
# rsqrt_61 => rsqrt_61
# to_129 => convert_element_type_323
# to_130 => convert_element_type_324
# where_1 => full_default_2, full_default_3
# where_61 => where_61
triton_per_fused__to_copy_add_clamp_mean_mul_neg_pow_rsqrt_scalar_tensor_where_18 = async_compile.triton('triton_per_fused__to_copy_add_clamp_mean_mul_neg_pow_rsqrt_scalar_tensor_where_18', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_mean_mul_neg_pow_rsqrt_scalar_tensor_where_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_mean_mul_neg_pow_rsqrt_scalar_tensor_where_18(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = 64504.0
    tmp5 = 65504.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = -tmp6
    tmp8 = triton_helpers.maximum(tmp1, tmp7)
    tmp9 = triton_helpers.minimum(tmp8, tmp6)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 768.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp11 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp17 * tmp24
    tmp26 = 0.03608439182435161
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp27, rmask)
''')
