

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/bb/cbbu5ixa2d2trv7dn45vtrncqtdvgthse7s3dlqh5yrcg4wwuyzl.py
# Source Nodes: [add_18, add_20, add_22, add_24, add_25, l__self___block_3_layer__1__dropout, l__self___block_4_layer_0_dropout, l__self___block_4_layer__1__dropout, l__self___block_5_layer_0_dropout, mean_11, mul_25, mul_26, pow_12, rsqrt_11], Original ATen: [aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_18 => add_23
# add_20 => add_26
# add_22 => add_28
# add_24 => add_31
# add_25 => add_32
# l__self___block_3_layer__1__dropout => clone_20
# l__self___block_4_layer_0_dropout => clone_23
# l__self___block_4_layer__1__dropout => clone_25
# l__self___block_5_layer_0_dropout => clone_28
# mean_11 => mean_11
# mul_25 => mul_25
# mul_26 => mul_26
# pow_12 => pow_12
# rsqrt_11 => rsqrt_11
triton_per_fused_add_clone_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused_add_clone_mean_mul_pow_rsqrt_13', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mean_mul_pow_rsqrt_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_mean_mul_pow_rsqrt_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp8, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp21, rmask)
''')
