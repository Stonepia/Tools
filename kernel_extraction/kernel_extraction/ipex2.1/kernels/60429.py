

# Original file: ./hf_T5_generate__45_inference_85.25/hf_T5_generate__45_inference_85.25_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/hh/chhnwx4tbwtrmobk7ppeahwt5fv45fgfjfgorfl3uyqkmz4lvtab.py
# Source Nodes: [add_15, add_18, add_20, add_21, l__self___decoder_block_1_layer_0_dropout, l__self___decoder_block_1_layer_1_dropout, l__self___decoder_block_1_layer__1__dropout, mean_6, mul_16, mul_17, pow_7, rsqrt_6], Original ATen: [aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# add_18 => add_22
# add_20 => add_24
# add_21 => add_25
# l__self___decoder_block_1_layer_0_dropout => clone_8
# l__self___decoder_block_1_layer_1_dropout => clone_10
# l__self___decoder_block_1_layer__1__dropout => clone_12
# mean_6 => mean_6
# mul_16 => mul_16
# mul_17 => mul_17
# pow_7 => pow_7
# rsqrt_6 => rsqrt_6
triton_per_fused_add_clone_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused_add_clone_mean_mul_pow_rsqrt_14', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mean_mul_pow_rsqrt_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_mean_mul_pow_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 512.0
    tmp14 = tmp11 / tmp13
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp19, rmask)
''')
