

# Original file: ./llama___60.0/llama___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ni/cni7nwl3yt3c4driuf57adv6x34jfocc2nocwiaiygwj45uhjip2.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1056
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 33
    x2 = xindex
    y1 = (yindex // 33)
    tmp8 = tl.load(in_ptr1 + (x2 + (512*y0) + (524288*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 33, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-512) + x2 + (512*y0) + (16384*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp10 = 0.3535533905932738
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (y0 + (33*x2) + (16896*y1)), tmp11, xmask & ymask)
''')
