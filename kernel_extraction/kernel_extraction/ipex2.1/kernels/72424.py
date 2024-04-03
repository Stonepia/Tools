

# Original file: ./YituTechConvBert__0_backward_243.1/YituTechConvBert__0_backward_243.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/he/chedgdzkcefoegq7dnrnqtl3n7y72kuaiaxgri6743rlr7blbmdh.py
# Source Nodes: [], Original ATen: [aten.col2im]

triton_poi_fused_col2im_18 = async_compile.triton('triton_poi_fused_col2im_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_col2im_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 55296
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 9
    y2 = (yindex // 3456)
    y5 = yindex % 3456
    y4 = (yindex // 9)
    tmp0 = tl.load(in_ptr0 + (x3 + (512*y0)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, YBLOCK])
    tmp5 = tl.load(in_ptr2 + (y5 + (3456*x3) + (1769472*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 520, tmp0)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 1, tmp3)
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp1 + (520*y4), [XBLOCK, YBLOCK])), tmp5, xmask)
''')
