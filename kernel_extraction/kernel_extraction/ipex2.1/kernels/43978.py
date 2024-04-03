

# Original file: ./yolov3___60.0/yolov3___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/sm/csmu5vkpraawfpes2a6yge4tkducxkde7e5wlx3zspnukranthpz.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_3
triton_poi_fused_cat_30 = async_compile.triton('triton_poi_fused_cat_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 391680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 85
    x1 = (xindex // 85) % 576
    x2 = (xindex // 48960)
    x3 = xindex % 48960
    tmp13 = tl.load(in_ptr0 + (x0 + (85*(x1 // 192)) + (255*(x1 % 192)) + (48960*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tmp3 & tmp2
    tmp5 = tl.load(in_ptr0 + (x0 + (85*(x1 // 192)) + (255*(x1 % 192)) + (48960*x2)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.where(tmp4, tmp5, 0.0)
    tmp7 = tl.load(in_ptr0 + (x0 + (85*(x1 // 192)) + (255*(x1 % 192)) + (48960*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tl.where(tmp2, tmp9, 0.0)
    tmp11 = tl.load(in_ptr0 + (x0 + (85*(x1 // 192)) + (255*(x1 % 192)) + (48960*x2)), tmp3 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp3, tmp11, 0.0)
    tmp14 = tl.where(tmp3, tmp12, tmp13)
    tmp15 = tl.where(tmp2, tmp10, tmp14)
    tl.store(out_ptr0 + (x3 + (1028160*x2)), tmp15, xmask)
''')