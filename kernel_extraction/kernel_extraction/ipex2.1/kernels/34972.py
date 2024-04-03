

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/mk/cmk2byp4e5dacuxu7q5bpcgqhyf6i5sehioqpf5dcg5vrvdkr7bl.py
# Source Nodes: [setitem_31, setitem_32, setitem_35], Original ATen: [aten.slice_scatter]
# setitem_31 => slice_scatter_155
# setitem_32 => slice_scatter_161
# setitem_35 => slice_scatter_178, slice_scatter_179
triton_poi_fused_slice_scatter_42 = async_compile.triton('triton_poi_fused_slice_scatter_42', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 21216)
    x1 = (xindex // 104) % 204
    x3 = xindex % 21216
    x4 = xindex
    tmp27 = tl.load(in_ptr3 + (x4), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = tl.load(in_ptr0 + ((-41808) + x3 + (20800*x2)), tmp10 & xmask, other=0.0)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = tmp5 & tmp5
    tmp14 = tl.load(in_ptr1 + ((-42432) + x4), tmp13 & xmask, other=0.0)
    tmp15 = tl.where(tmp13, tmp14, 0.0)
    tmp16 = tl.load(in_ptr2 + ((-42432) + x4), tmp13 & xmask, other=0.0)
    tmp17 = tl.where(tmp13, tmp16, 0.0)
    tmp18 = tl.load(in_ptr3 + (x4), tmp5 & xmask, other=0.0)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.where(tmp5, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp12, tmp20)
    tmp22 = tl.where(tmp5, tmp21, 0.0)
    tmp23 = tl.load(in_ptr1 + ((-42432) + x4), tmp5 & xmask, other=0.0)
    tmp24 = tl.where(tmp5, tmp23, 0.0)
    tmp25 = tl.load(in_ptr2 + ((-42432) + x4), tmp5 & xmask, other=0.0)
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tmp28 = tl.where(tmp5, tmp26, tmp27)
    tmp29 = tl.where(tmp5, tmp24, tmp28)
    tmp30 = tl.where(tmp5, tmp22, tmp29)
    tl.store(out_ptr0 + (x4), tmp30, xmask)
''')
