

# Original file: ./visformer_small___60.0/visformer_small___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/4f/c4f5kgxw6xqrmwt3h23jukjtieumvphpipcpa2ljjs7sw7emhhjr.py
# Source Nodes: [getattr_l__self___stage3___0___attn_proj, reshape_9], Original ATen: [aten.clone, aten.convolution]
# getattr_l__self___stage3___0___attn_proj => _convolution_pointwise_default_14
# reshape_9 => clone_53
triton_poi_fused_clone_convolution_29 = async_compile.triton('triton_poi_fused_clone_convolution_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072, 64], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_convolution_29(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    y4 = yindex % 768
    y5 = (yindex // 768)
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (6272*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr1 + (y4 + (768*x2) + (37632*y5)), tmp0, xmask)
''')
