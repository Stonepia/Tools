

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/4q/c4qh73pvv6t242y4ue6nzx2um333zpbz5rxomyf5vt72pdslmknl.py
# Source Nodes: [l__mod___cell_4_comb_iter_0_left_act_1, pad_18], Original ATen: [aten.constant_pad_nd, aten.relu]
# l__mod___cell_4_comb_iter_0_left_act_1 => relu_87
# pad_18 => constant_pad_nd_20
triton_poi_fused_constant_pad_nd_relu_45 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_45', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13996800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19440) % 45
    x1 = (xindex // 432) % 45
    x3 = (xindex // 874800)
    x4 = xindex % 19440
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-18576) + x4 + (18144*x2) + (762048*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.where(tmp10, tmp12, 0.0)
    tl.store(out_ptr0 + (x6), tmp13, xmask)
''')