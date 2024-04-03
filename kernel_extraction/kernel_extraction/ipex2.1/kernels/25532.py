

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/4d/c4dt4y3vvmlpy7dwfgxv6bg5dtoletoocpjfbhlo2rjra2ke4fte.py
# Source Nodes: [add_217, add_218, l__mod___mask_decoder_output_upscaling_0, l__mod___mask_decoder_output_upscaling_2, l__mod___mask_decoder_output_upscaling_3, mean_4, mean_5, mul_164, pow_3, sqrt_2, sub_72, truediv_12], Original ATen: [aten.add, aten.convolution, aten.div, aten.gelu, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_217 => add_398
# add_218 => add_399
# l__mod___mask_decoder_output_upscaling_0 => convolution_3
# l__mod___mask_decoder_output_upscaling_2 => add_400, convert_element_type_354, convert_element_type_355, erf_32, mul_407, mul_408, mul_409
# l__mod___mask_decoder_output_upscaling_3 => convolution_4
# mean_4 => mean_4
# mean_5 => mean_5
# mul_164 => mul_406
# pow_3 => pow_3
# sqrt_2 => sqrt_2
# sub_72 => sub_184
# truediv_12 => div_51
triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_57 = async_compile.triton('triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_57', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_57', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = 64.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp17 / tmp8
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = tmp11 / tmp23
    tmp25 = tmp18 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 0.5
    tmp30 = tmp28 * tmp29
    tmp31 = 0.7071067811865476
    tmp32 = tmp28 * tmp31
    tmp33 = libdevice.erf(tmp32)
    tmp34 = 1.0
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp37, rmask)
''')
