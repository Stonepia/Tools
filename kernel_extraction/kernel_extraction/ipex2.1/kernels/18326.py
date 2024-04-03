

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/k4/ck4tqizsuupr4oozlkoxuavizhmphmufwodv5enn7x7cd2mwepwa.py
# Source Nodes: [l__mod___cell_5_conv_prev_1x1_path_1_avgpool], Original ATen: [aten.avg_pool2d]
# l__mod___cell_5_conv_prev_1x1_path_1_avgpool => avg_pool2d_4
triton_poi_fused_avg_pool2d_39 = async_compile.triton('triton_poi_fused_avg_pool2d_39', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7620480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1080
    x1 = (xindex // 1080) % 21
    x2 = (xindex // 22680)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2160*x1) + (90720*x2)), xmask).to(tl.float32)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')
