

# Original file: ./XLNetLMHeadModel__0_backward_567.1/XLNetLMHeadModel__0_backward_567.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ym/cymo3rxrk5kqyjqhh7ub2zd7crqkjsbprxe43oqv4svqwpcwyejh.py
# Source Nodes: [], Original ATen: [aten.bmm]

triton_poi_fused_bmm_16 = async_compile.triton('triton_poi_fused_bmm_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bmm_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 524288
    x1 = (xindex // 524288)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (524288*(x1 % 16)) + (8388608*(((x0 + (524288*(x1 % 16)) + (8388608*(x1 // 16))) // 8388608) % 8))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''')
