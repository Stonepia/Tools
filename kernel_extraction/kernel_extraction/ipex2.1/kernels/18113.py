

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/p7/cp7gop7ap75wno7j6wkdop3e7g2laqx63orvt7v6msjwqw5a5zvq.py
# Source Nodes: [l__mod___cell_0_conv_prev_1x1_path_2_avgpool, l__mod___cell_0_conv_prev_1x1_path_2_pad], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd]
# l__mod___cell_0_conv_prev_1x1_path_2_avgpool => avg_pool2d_3
# l__mod___cell_0_conv_prev_1x1_path_2_pad => constant_pad_nd_19
triton_poi_fused_avg_pool2d_constant_pad_nd_21 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7620480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 11340) % 42
    x1 = (xindex // 270) % 42
    x0 = xindex % 270
    x3 = (xindex // 476280)
    x6 = xindex
    tmp0 = 1 + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (22680 + x0 + (540*x1) + (44820*x2) + (1860030*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x6), tmp14, xmask)
''')
