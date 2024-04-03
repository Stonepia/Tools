

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/hf/chf575bgwgzi2pavk6jpvhofq2y5kt4igg3mnszr665moul7y2zr.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_15 = async_compile.triton('triton_poi_fused_add_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 24624)
    x0 = xindex % 513
    x1 = (xindex // 513) % 48
    x3 = xindex
    tmp12 = tl.load(in_ptr0 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*((((12*((((12*(x1 // 12)) + (x1 % 12)) // 12) % 4)) + (x1 % 12)) // 12) % 4))), None)
    tmp0 = x2
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, 0.0)
    tmp9 = tl.load(in_ptr0 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*((((12*((((12*(x1 // 12)) + (x1 % 12)) // 12) % 4)) + (x1 % 12)) // 12) % 4))), tmp2, other=0.0)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tl.where(tmp2, tmp10, 0.0)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + ((-197632) + x0 + (257*x2)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp15 = (tmp14 != 0)
    tmp16 = tl.load(in_ptr0 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*((((12*((((12*(x1 // 12)) + (x1 % 12)) // 12) % 4)) + (x1 % 12)) // 12) % 4))), tmp6, other=0.0)
    tmp17 = tl.where(tmp15, tmp7, tmp16)
    tmp18 = tl.where(tmp6, tmp17, 0.0)
    tmp19 = tl.where(tmp5, tmp18, tmp7)
    tmp20 = tl.where(tmp2, tmp19, 0.0)
    tmp21 = tl.where(tmp2, tmp20, tmp7)
    tmp22 = tmp13 + tmp21
    tl.store(out_ptr0 + (x3), tmp22, None)
''')
