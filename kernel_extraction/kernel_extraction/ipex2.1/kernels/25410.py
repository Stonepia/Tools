

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zt/cztegxwcindx6e2rrt3ozs5u5ywnejmpvqx5l3caqm2vfe3xjhk6.py
# Source Nodes: [add_217, add_218, l__mod___mask_decoder_output_upscaling_0, l__mod___mask_decoder_output_upscaling_2, l__mod___mask_decoder_output_upscaling_3, mean_4, mean_5, mul_164, pow_3, sqrt_2, sub_72, truediv_12], Original ATen: [aten.add, aten.convolution, aten.div, aten.gelu, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_217 => add_398
# add_218 => add_399
# l__mod___mask_decoder_output_upscaling_0 => convolution_3
# l__mod___mask_decoder_output_upscaling_2 => add_400, erf_32, mul_407, mul_408, mul_409
# l__mod___mask_decoder_output_upscaling_3 => convolution_4
# mean_4 => mean_4
# mean_5 => mean_5
# mul_164 => mul_406
# pow_3 => pow_3
# sqrt_2 => sqrt_2
# sub_72 => sub_184
# truediv_12 => div_51
triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_64 = async_compile.triton('triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_64', '''
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
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_64', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tmp14 / tmp7
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.sqrt(tmp18)
    tmp20 = tmp9 / tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = 0.7071067811865476
    tmp27 = tmp23 * tmp26
    tmp28 = libdevice.erf(tmp27)
    tmp29 = 1.0
    tmp30 = tmp28 + tmp29
    tmp31 = tmp25 * tmp30
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp31, rmask)
''')
