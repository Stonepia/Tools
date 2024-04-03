

# Original file: ./sam___60.0/sam___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/7r/c7r2bzeaef7xu6mhn6oliwjsouh2b7kve5nz2mo4b3kkjxnwm7b6.py
# Source Nodes: [l__mod___image_encoder_patch_embed_proj, pad, sub, truediv], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.div, aten.sub]
# l__mod___image_encoder_patch_embed_proj => _convolution_pointwise_default_2
# pad => constant_pad_nd
# sub => sub
# truediv => div
triton_poi_fused_constant_pad_nd_convolution_div_sub_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_convolution_div_sub_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4, 1048576], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_div_sub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_convolution_div_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 1024)
    x1 = xindex % 1024
    y0 = yindex
    x3 = xindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x1 + (256*x2) + (65536*y0)), tmp5 & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp5 & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp5 & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 / tmp9
    tmp11 = tl.where(tmp5, tmp10, 0.0)
    tl.store(out_ptr1 + (y0 + (3*x3)), tmp11, ymask)
''')
