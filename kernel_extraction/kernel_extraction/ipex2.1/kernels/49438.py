

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/k2/ck2wa6xmaorlfzqxh5e6ghwt2eensfakw33juasb5e7psqpx276c.py
# Source Nodes: [scaled_dot_product_attention], Original ATen: [aten.clone]
# scaled_dot_product_attention => clone_266
triton_poi_fused_clone_27 = async_compile.triton('triton_poi_fused_clone_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3014400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 785
    x2 = (xindex // 37680) % 16
    x3 = (xindex // 602880)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (768*x1) + (602880*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')
