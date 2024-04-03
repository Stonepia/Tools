

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/j2/cj2qohqk674y7qycs2p4x3owro6hlucofq3bdqebppqa7j56tdi3.py
# Source Nodes: [pad_2], Original ATen: [aten.constant_pad_nd]
# pad_2 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_22 = async_compile.triton('triton_poi_fused_constant_pad_nd_22', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 1536
    x0 = xindex % 64
    x2 = (xindex // 98304)
    x3 = xindex
    tmp0 = (-256) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-786432) + x0 + (64*x2) + (3072*x1)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0 + (64*(x2 % 12))), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.where(tmp5, tmp8, -1.0)
    tl.store(out_ptr0 + (x3), tmp9, None)
''')