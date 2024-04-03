

# Original file: ./sam___60.0/sam___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/57/c5777nw3ufoxn3qeattqu6hpustbh7uniy6b73wfineanah4lw47.py
# Source Nodes: [add_217, add_218, l__self___mask_decoder_output_upscaling_0, l__self___mask_decoder_output_upscaling_2, l__self___mask_decoder_output_upscaling_3, mean_4, mean_5, mul_164, pow_3, sqrt_2, sub_72, truediv_12], Original ATen: [aten._to_copy, aten.add, aten.convolution, aten.div, aten.gelu, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_217 => add_398
# add_218 => add_399
# l__self___mask_decoder_output_upscaling_0 => convert_element_type_697, convolution_3
# l__self___mask_decoder_output_upscaling_2 => add_400, erf_32, mul_407, mul_408, mul_409
# l__self___mask_decoder_output_upscaling_3 => convert_element_type_701, convolution_4
# mean_4 => mean_4
# mean_5 => mean_5
# mul_164 => mul_406
# pow_3 => convert_element_type_700, pow_3
# sqrt_2 => sqrt_2
# sub_72 => sub_184
# truediv_12 => div_51
triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_62 = async_compile.triton('triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_62', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_62', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_convolution_div_gelu_mean_mul_pow_sqrt_sub_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = 64.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = tmp17 / tmp8
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sqrt(tmp21)
    tmp23 = tmp12 / tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = 0.5
    tmp28 = tmp26 * tmp27
    tmp29 = 0.7071067811865476
    tmp30 = tmp26 * tmp29
    tmp31 = libdevice.erf(tmp30)
    tmp32 = 1.0
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr3 + (r1 + (64*x0)), tmp35, rmask)
''')
