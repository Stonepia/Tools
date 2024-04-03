

# Original file: ./hf_T5_generate__56_inference_96.36/hf_T5_generate__56_inference_96.36_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/d5/cd5g434ztntwyjmxsmrj2lemcgy2vdsdcihsupqtqxeoqgpgvxz4.py
# Source Nodes: [add_15, add_18, add_20, add_23, add_24, l__self___decoder_block_1_layer_0_dropout, l__self___decoder_block_1_layer_1_dropout, l__self___decoder_block_1_layer__1__dropout, l__self___decoder_block_2_layer_0_dropout, mean_7, mul_18, mul_19, pow_8, rsqrt_7], Original ATen: [aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# add_18 => add_22
# add_20 => add_24
# add_23 => add_27
# add_24 => add_28
# l__self___decoder_block_1_layer_0_dropout => clone_8
# l__self___decoder_block_1_layer_1_dropout => clone_10
# l__self___decoder_block_1_layer__1__dropout => clone_12
# l__self___decoder_block_2_layer_0_dropout => clone_14
# mean_7 => mean_7
# mul_18 => mul_18
# mul_19 => mul_19
# pow_8 => pow_8
# rsqrt_7 => rsqrt_7
triton_per_fused_add_clone_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_per_fused_add_clone_mean_mul_pow_rsqrt_15', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mean_mul_pow_rsqrt_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_mean_mul_pow_rsqrt_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
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
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp21, rmask)
''')
