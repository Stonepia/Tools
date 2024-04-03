

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/pc/cpc2vrtwelx7n5chmaumss2a4gudjl4gxdfmbvuctia37qp6ikh5.py
# Source Nodes: [add_193, add_194, l__self___image_encoder_neck_2, mean, mean_1, mul_160, pow_1, sqrt, sub_65, truediv_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_193 => add_353
# add_194 => add_354
# l__self___image_encoder_neck_2 => convert_element_type_582
# mean => mean
# mean_1 => mean_1
# mul_160 => mul_384
# pow_1 => convert_element_type_581, pow_1
# sqrt => sqrt
# sub_65 => sub_161
# truediv_1 => div_33
triton_per_fused__to_copy_add_div_mean_mul_pow_sqrt_sub_40 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_pow_sqrt_sub_40', '''
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_pow_sqrt_sub_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_pow_sqrt_sub_40(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 / tmp6
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = tmp10 / tmp20
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp25, rmask)
''')
