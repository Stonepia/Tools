

# Original file: ./visformer_small___60.0/visformer_small___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/m4/cm4fjyy6ulerh2o2lae5g7ilt277ni2bwvmyz3o6tg5aenhlshqm.py
# Source Nodes: [getattr_l__mod___stage2___0___attn_proj, reshape_1], Original ATen: [aten.clone, aten.convolution]
# getattr_l__mod___stage2___0___attn_proj => _convolution_pointwise_default_31
# reshape_1 => clone_20
triton_poi_fused_clone_convolution_15 = async_compile.triton('triton_poi_fused_clone_convolution_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_convolution_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_convolution_15(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 49152
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    y3 = yindex
    y4 = yindex % 384
    y5 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (12544*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr1 + (y4 + (384*x2) + (75264*y5)), tmp0, xmask)
''')
