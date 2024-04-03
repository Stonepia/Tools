

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/g4/cg4ohwp3hxocvi7gq5v3rlp7tagqhwe7z6yika2hacbmhp5ywgnl.py
# Source Nodes: [max_pool2d_1, pad_1], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# max_pool2d_1 => max_pool2d_with_indices_1
# pad_1 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_28 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y0 = yindex % 512
    y1 = (yindex // 512)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (512 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1024 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (14848 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (15360 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (15872 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (29696 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (30208 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (30720 + y0 + (1024*x2) + (29696*x3) + (430592*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x5 + (196*y4)), tmp16, xmask)
''')
