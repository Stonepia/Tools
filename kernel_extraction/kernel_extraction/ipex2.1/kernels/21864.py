

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/f7/cf7myeivzdhgo442gwcxx7y3lcpfkbnsuejv6564kihkfecpoahc.py
# Source Nodes: [l__self___stage4_0_fuse_act_1], Original ATen: [aten.relu]
# l__self___stage4_0_fuse_act_1 => relu_181
triton_poi_fused_relu_8 = async_compile.triton('triton_poi_fused_relu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_relu_8(in_out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 36
    y1 = (yindex // 36)
    tmp0 = tl.load(in_out_ptr0 + (y0 + (36*x2) + (28224*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (36*x2) + (28224*y1)), tmp1, xmask & ymask)
''')
