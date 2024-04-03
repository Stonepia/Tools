

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/3y/c3ynyortnwnfdktpp5c2kk76y3gk3s7oot55hcji7542c2n4oxo6.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_35 = async_compile.triton('triton_poi_fused_add_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768) % 4
    x2 = (xindex // 3072)
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + (768*x2) + (786432*x1)), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + (x0 + (768*x2) + (786432*x1)), tmp6, None)
''')
