

# Original file: ./tacotron2__26_inference_66.6/tacotron2__26_inference_66.6_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/bc/cbc3wnbelwh32nyopn36bumim5nlszhvkbvffpucorjn3ytvbcwh.py
# Source Nodes: [add, getattr_l__self___postnet_convolutions_0___0___conv, getattr_l__self___postnet_convolutions_1___0___conv, getattr_l__self___postnet_convolutions_2___0___conv, getattr_l__self___postnet_convolutions_3___0___conv, getattr_l__self___postnet_convolutions__1____0___conv, tanh, tanh_1, tanh_2, tanh_3], Original ATen: [aten.add, aten.convolution, aten.tanh]
# add => add_10
# getattr_l__self___postnet_convolutions_0___0___conv => convolution_default_19
# getattr_l__self___postnet_convolutions_1___0___conv => convolution_default_18
# getattr_l__self___postnet_convolutions_2___0___conv => convolution_default_17
# getattr_l__self___postnet_convolutions_3___0___conv => convolution_default_16
# getattr_l__self___postnet_convolutions__1____0___conv => convolution_default_15
# tanh => tanh
# tanh_1 => tanh_1
# tanh_2 => tanh_2
# tanh_3 => tanh_3
triton_poi_fused_add_convolution_tanh_2 = async_compile.triton('triton_poi_fused_add_convolution_tanh_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_tanh_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_convolution_tanh_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 54848
    xnumel = 80
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 857
    y1 = (yindex // 857)
    tmp0 = tl.load(in_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0 + (857*x2) + (68560*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2 + (80*y3)), tmp4, xmask & ymask)
''')
