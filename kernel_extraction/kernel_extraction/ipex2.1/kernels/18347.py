

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/zi/czib2pwj2n4jgkj3au3w3nxfqw5xbydslpo2qelz34tti6vod644.py
# Source Nodes: [cat_22, l__mod___cell_8_comb_iter_3_left_act_1], Original ATen: [aten.cat, aten.relu]
# cat_22 => cat_13
# l__mod___cell_8_comb_iter_3_left_act_1 => relu_152
triton_poi_fused_cat_relu_60 = async_compile.triton('triton_poi_fused_cat_relu_60', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_60', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_relu_60(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1672704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 864
    x2 = (xindex // 864)
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr1 + (x1 + (4320*x2)), tmp0, xmask)
''')
