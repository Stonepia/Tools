

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/nm/cnmd3f6hiqpi7b7d7gydtzctaak6mcc7we7e7gny5xaahd3vqjus.py
# Source Nodes: [l__mod___cell_9_conv_prev_1x1_path_2_avgpool, l__mod___cell_9_conv_prev_1x1_path_2_pad], Original ATen: [aten.avg_pool2d, aten.constant_pad_nd]
# l__mod___cell_9_conv_prev_1x1_path_2_avgpool => avg_pool2d_7
# l__mod___cell_9_conv_prev_1x1_path_2_pad => constant_pad_nd_39
triton_poi_fused_avg_pool2d_constant_pad_nd_53 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_53', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_53', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4181760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 23760) % 11
    x1 = (xindex // 2160) % 11
    x0 = xindex % 2160
    x3 = (xindex // 261360)
    x6 = xindex
    tmp0 = 1 + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (47520 + x0 + (4320*x1) + (90720*x2) + (952560*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x6), tmp14, xmask)
''')
