

# Original file: ./YituTechConvBert__0_forward_241.0/YituTechConvBert__0_forward_241.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/op/copnat6ogavsr4owq7pd6dul2zzzfq4wtef27fh4xsvszgq4yrbj.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 6
    x3 = (xindex // 196608)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (384*x1) + (196608*x3)), None)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')
