

# Original file: ./DALLE2_pytorch__42_inference_82.22/DALLE2_pytorch__42_inference_82.22.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zb/czbqsyxla5wsjrqc2uojhg2qosciwyanjwd7kn4d4areue65h46x.py
# Source Nodes: [l__self___init_conv_convs_0, l__self___init_conv_convs_1, l__self___init_conv_convs_2], Original ATen: [aten.convolution]
# l__self___init_conv_convs_0 => _convolution_pointwise_default_2
# l__self___init_conv_convs_1 => _convolution_pointwise_default_1
# l__self___init_conv_convs_2 => _convolution_pointwise_default
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4, 16384], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + (16384*y0)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x1)), tmp0, ymask)
    tl.store(out_ptr1 + (y0 + (3*x1)), tmp0, ymask)
    tl.store(out_ptr2 + (y0 + (3*x1)), tmp0, ymask)
''')
