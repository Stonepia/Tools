

# Original file: ./volo_d1_224___60.0/volo_d1_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/3r/c3r4rcuu6bi6332c3nknxmet2djxeewtq66pu4j542reve6amghc.py
# Source Nodes: [fold], Original ATen: [aten.col2im]
# fold => _unsafe_index_put, full_default
triton_poi_fused_col2im_7 = async_compile.triton('triton_poi_fused_col2im_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_col2im_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 3
    x3 = (xindex // 3) % 14
    x4 = (xindex // 42) % 3
    x5 = (xindex // 126)
    y0 = yindex % 192
    y1 = (yindex // 192)
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + ((32*x2) + (96*x4) + (288*x3) + (4032*x5) + (56448*((x2 + (3*x4) + (9*y0)) // 288)) + (338688*y1) + (((x2 + (3*x4) + (9*y0)) // 9) % 32)), xmask, eviction_policy='evict_last')
    tl.atomic_add(out_ptr0 + (x2 + (2*x3) + (30*x4) + (60*x5) + (900*y6)), tmp0, xmask)
''')
