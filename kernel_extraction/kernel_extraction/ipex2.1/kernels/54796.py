

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/dk/cdkfs6quqheawd3lq6rq3crr3z6depiu3lyupc5lywilnmnsmdsy.py
# Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# pad_3 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_23 = async_compile.triton('triton_poi_fused_constant_pad_nd_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i1', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37847040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x1 = (xindex // 770) % 48
    x2 = (xindex // 36960)
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*(x1 // 12))), tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2 + (1024*(x1 // 12))), tmp2, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*(x1 // 12))), tmp2, other=0.0)
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp7, tmp6)
    tmp9 = tmp4 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp2, tmp11, 0.0)
    tl.store(out_ptr0 + (x0 + (770*x2) + (788480*x1)), tmp12, None)
''')