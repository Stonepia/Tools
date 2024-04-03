

# Original file: ./DALLE2_pytorch__35_inference_75.15/DALLE2_pytorch__35_inference_75.15.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/po/cpojqns5kosql7z6xnqx4ligcq22fkjmkmmzrzaxxz24na36pgh2.py
# Source Nodes: [add, amax, mean, mul, mul_1, rsqrt, sub, truediv, var], Original ATen: [aten.add, aten.amax, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sub, aten.var]
# add => add
# amax => amax
# mean => mean
# mul => mul
# mul_1 => mul_1
# rsqrt => rsqrt
# sub => sub
# truediv => div
# var => var
triton_per_fused_add_amax_div_mean_mul_rsqrt_sub_var_0 = async_compile.triton('triton_per_fused_add_amax_div_mean_mul_rsqrt_sub_var_0', '''
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_amax_div_mean_mul_rsqrt_sub_var_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_amax_div_mean_mul_rsqrt_sub_var_0(in_ptr0, in_ptr1, out_ptr3, xnumel, rnumel):
    xnumel = 520
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp3, 0))
    tmp5 = tmp0 / tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tl.full([1], 512, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tmp18 = tmp6 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = 512.0
    tmp25 = tmp9 / tmp24
    tmp26 = tmp5 - tmp25
    tmp27 = tmp23 / tmp24
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp26 * tmp30
    tmp33 = tmp31 * tmp32
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp33, rmask & xmask)
''')
