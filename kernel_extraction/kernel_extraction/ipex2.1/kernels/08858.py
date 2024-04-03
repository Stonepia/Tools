

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/eb/cebs33q2b7jcxvvlewqpwy6rzqwisqi7ngdlfcncgfmesljr4udd.py
# Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.clone, aten.mul]
# scaled_dot_product_attention => clone_2, mul_3
triton_poi_fused_clone_mul_3 = async_compile.triton('triton_poi_fused_clone_mul_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 256], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 16
    y2 = (yindex // 512) % 4
    y3 = (yindex // 2048)
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + y0 + (32*y2) + (384*x4) + (75264*y1) + (1204224*y3)), xmask, eviction_policy='evict_last')
    tmp1 = 0.42044820762685725
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4 + (196*y5)), tmp2, xmask)
''')
