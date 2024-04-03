

# Original file: ./eca_halonext26ts___60.0/eca_halonext26ts___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/wr/cwrjcxjp4rxs5x5j5lg43w3lwjvl7i2mrtufz6r3rtnszjw5akkf.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_3
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = (yindex // 16) % 4
    y0 = yindex % 16
    y2 = (yindex // 64)
    y4 = yindex
    tmp0 = (-2) + (8*((x3 + (144*y1)) // 288)) + (x3 // 12)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + (8*(y1 % 2)) + (x3 % 12)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-13056) + (384*(x3 % 12)) + (3072*(y1 % 2)) + (6144*(x3 // 12)) + (49152*((x3 + (144*y1)) // 288)) + (98304*((x3 + (144*y1) + (576*y0) + (27648*y2)) // 221184)) + (((x3 + (144*y1) + (576*y0) + (27648*y2)) // 576) % 384)), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tl.store(out_ptr0 + (x3 + (144*y4)), tmp12, xmask)
''')
