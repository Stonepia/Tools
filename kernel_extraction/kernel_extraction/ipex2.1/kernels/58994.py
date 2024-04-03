

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/d3/cd3ldnlaj2gcvjgfluysgszju5vljwayvzgdefepwszts4bo72au.py
# Source Nodes: [], Original ATen: [aten.clone, aten.slice_backward]

triton_poi_fused_clone_slice_backward_18 = async_compile.triton('triton_poi_fused_clone_slice_backward_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_backward_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_slice_backward_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6279120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x2 = (xindex // 24624)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 258, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 1 + x2
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = (-257) + x0
    tmp8 = tl.full([1], 257, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = 0.0
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = tl.load(in_ptr0 + (24367 + x5), tmp6 & xmask, other=0.0)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp6, tmp14, 0.0)
    tmp16 = tl.load(in_ptr0 + (24367 + x5), tmp2 & xmask, other=0.0)
    tmp17 = tl.where(tmp5, tmp15, tmp16)
    tmp18 = tl.load(in_ptr1 + (x0 + (257*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = (tmp18 != 0)
    tmp20 = tl.load(in_ptr0 + (24367 + x5), tmp10 & xmask, other=0.0)
    tmp21 = tl.where(tmp19, tmp11, tmp20)
    tmp22 = tl.where(tmp10, tmp21, 0.0)
    tmp23 = tl.where(tmp9, tmp22, tmp11)
    tmp24 = tl.where(tmp6, tmp23, 0.0)
    tmp25 = tl.where(tmp5, tmp24, tmp11)
    tmp26 = tmp17 + tmp25
    tmp27 = tl.where(tmp2, tmp26, 0.0)
    tmp28 = tl.where(tmp2, tmp27, tmp11)
    tl.store(out_ptr0 + (x5), tmp28, xmask)
''')
