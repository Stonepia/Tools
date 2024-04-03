

# Original file: ./convit_base___60.0/convit_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/oa/coatblzjyfoe5yeub7smorzic5shwyq3ded3qknc5l6uoy6lzoqy.py
# Source Nodes: [add_1, itruediv, matmul_1, mul, mul_1, softmax, sum_1], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.div, aten.mul, aten.sum]
# add_1 => add_3
# itruediv => div_2
# matmul_1 => convert_element_type_14
# mul => mul_2
# mul_1 => mul_3
# softmax => amax, convert_element_type_8, convert_element_type_9, div, exp, sub_1, sum_1
# sum_1 => sum_3
triton_per_fused__softmax__to_copy_add_div_mul_sum_4 = async_compile.triton('triton_per_fused__softmax__to_copy_add_div_mul_sum_4', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_div_mul_sum_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_div_mul_sum_4(in_ptr0, in_ptr1, in_ptr2, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = 0.14433756729740643
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tmp9 / tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp14 * tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp20 / tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr4 + (r1 + (196*x0)), tmp26, rmask)
''')
