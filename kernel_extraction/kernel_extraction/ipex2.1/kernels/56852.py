

# Original file: ./YituTechConvBert__0_backward_207.1/YituTechConvBert__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/3y/c3ybrlkuko6wb7zaionomebrjf5txtr2fqnczcu5frqnrgorcxxw.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_32 = async_compile.triton('triton_poi_fused_view_32', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_view_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 54
    x1 = (xindex // 54)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((9*(((x0 + (54*(x1 % 512)) + (27648*(x1 // 512))) // 9) % 49152)) + (x0 % 9)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')