

# Original file: ./tf_mixnet_l___60.0/tf_mixnet_l___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/eq/ceqiu277whhjns3enjfu2pxjaor5n5hor6br5febfhbgqdhpmdrh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
# getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0 => _convolution_pointwise_default_92
triton_poi_fused_convolution_61 = async_compile.triton('triton_poi_fused_convolution_61', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_61', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_61(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 39936
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 312
    y1 = (yindex // 312)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (122304*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (312*x2) + (61152*y1)), tmp0, xmask)
''')