

# Original file: ./pit_b_224___60.0/pit_b_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/5t/c5ti2m2vac3gwond5cupkgvvth2mmmn4nfrgkhnxk3omoy5mzekf.py
# Source Nodes: [scaled_dot_product_attention_3], Original ATen: [aten.clone, aten.mul]
# scaled_dot_product_attention_3 => clone_23, mul_30
triton_poi_fused_clone_mul_19 = async_compile.triton('triton_poi_fused_clone_mul_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 512], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_mul_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (512 + y0 + (1536*x2) + (394752*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + (257*y3)), tmp2, xmask)
''')
