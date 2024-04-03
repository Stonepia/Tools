

# Original file: ./convit_base___60.0/convit_base___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/te/ctedze72j7e5xhbcp2zmxspmh5mnapqptaldkzwcbskpdb2a4cs5.py
# Source Nodes: [add_1, itruediv, mul, mul_1, softmax, sum_1], Original ATen: [aten._softmax, aten.add, aten.div, aten.mul, aten.sum]
# add_1 => add_4
# itruediv => div_2
# mul => mul_2
# mul_1 => mul_3
# softmax => amax, div, exp, sub_1, sum_1
# sum_1 => sum_3
triton_per_fused__softmax_add_div_mul_sum_4 = async_compile.triton('triton_per_fused__softmax_add_div_mul_sum_4', '''
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
    size_hints=[262144, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_div_mul_sum_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_div_mul_sum_4(in_ptr0, in_ptr1, in_ptr2, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 196) % 16
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = 0.14433756729740643
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tmp8 / tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tmp17 / tmp21
    tl.store(out_ptr4 + (r1 + (196*x0)), tmp22, rmask)
''')
