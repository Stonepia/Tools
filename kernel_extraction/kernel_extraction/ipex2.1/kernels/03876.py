

# Original file: ./vision_maskrcnn__32_inference_72.12/vision_maskrcnn__32_inference_72.12_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/i5/ci5raqpa3vegwg2n65medxjgoefwgipna7i6c2lxpanmzdo2ytat.py
# Source Nodes: [full_like], Original ATen: [aten.full_like]
# full_like => full
triton_poi_fused_full_like_0 = async_compile.triton('triton_poi_fused_full_like_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_like_0', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_full_like_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (5*x0), tmp0, xmask)
''')
