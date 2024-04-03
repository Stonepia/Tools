

# Original file: ./shufflenet_v2_x1_0___60.0/shufflenet_v2_x1_0___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/nr/cnrcr5ycjwnfdofzvbfvj2sq7vnhuqgyg6gv4z3z6bkyltbuzjhi.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_0, getattr_l__mod___stage4___0___branch2_2], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___stage4___0___branch1_0 => _convolution_pointwise_default_14
# getattr_l__mod___stage4___0___branch2_2 => fn_8
triton_poi_fused_convolution_relu_13 = async_compile.triton('triton_poi_fused_convolution_relu_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_relu_13(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 14848
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (232*x2) + (45472*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (232*x2) + (45472*y1)), tmp0, xmask & ymask)
''')
