

# Original file: ./cm3leon_generate__26_inference_66.6/cm3leon_generate__26_inference_66.6_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/3b/c3bk4nrnsjvscpehygqeqz4wubgl4vy7c7vownklvrc45epo5bh3.py
# Source Nodes: [iadd_5, nan_to_num, nan_to_num_4, softmax_4, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_5 => add_32
# nan_to_num => full_default_3, full_default_4
# nan_to_num_4 => eq_8, eq_9, isnan_4, where_13, where_14, where_15
# softmax_4 => amax_4, div_4, exp_4, sub_14, sum_5
# triu => full_default_1
triton_per_fused__softmax_add_nan_to_num_triu_24 = async_compile.triton('triton_per_fused__softmax_add_nan_to_num_triu_24', '''
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
    size_hints=[16, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_nan_to_num_triu_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_nan_to_num_triu_24(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 29
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (29*x0)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp1 = float("inf")
    tmp2 = tmp0 == tmp1
    tmp3 = float("-inf")
    tmp4 = tmp0 == tmp3
    tmp5 = libdevice.isnan(tmp0).to(tl.int1)
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = -3.4028234663852886e+38
    tmp9 = tl.where(tmp4, tmp8, tmp7)
    tmp10 = 3.4028234663852886e+38
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp14 = tmp11 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, float("-inf"))
    tmp18 = triton_helpers.max2(tmp17, 1)[:, None]
    tmp19 = tmp14 - tmp18
    tmp20 = tl.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp20 / tmp24
    tl.store(out_ptr2 + (r1 + (29*x0)), tmp25, rmask & xmask)
''')
