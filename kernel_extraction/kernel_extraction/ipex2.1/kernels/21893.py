

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/oe/coen3jzg4zgof74l6ldpm5t7gbkvylxhij4joyikty47kvcimblz.py
# Source Nodes: [l__mod___stage3_0_fuse_act], Original ATen: [aten.relu]
# l__mod___stage3_0_fuse_act => relu_59
triton_poi_fused_relu_3 = async_compile.triton('triton_poi_fused_relu_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_3(in_out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 18
    y1 = (yindex // 18)
    tmp0 = tl.load(in_out_ptr0 + (y0 + (18*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (18*x2) + (56448*y1)), tmp1, xmask & ymask)
''')
