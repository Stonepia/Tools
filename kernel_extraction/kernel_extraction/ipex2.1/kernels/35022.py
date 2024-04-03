

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/rc/crcz2rlanqksnlwc75q6sgzpqvsq5qetofj5v5axx3w76dzl3vbb.py
# Source Nodes: [setitem_31, setitem_32], Original ATen: [aten.slice_scatter]
# setitem_31 => slice_scatter_154, slice_scatter_155
# setitem_32 => slice_scatter_160, slice_scatter_161
triton_poi_fused_slice_scatter_39 = async_compile.triton('triton_poi_fused_slice_scatter_39', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_39', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 21216)
    x1 = (xindex // 104) % 204
    x3 = xindex % 21216
    x4 = xindex
    tmp28 = tl.load(in_ptr2 + (x4), xmask).to(tl.float32)
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
    tmp11 = tl.load(in_ptr0 + ((-41808) + x3 + (20800*x2)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = tmp5 & tmp5
    tmp14 = tmp9 & tmp13
    tmp15 = tl.load(in_ptr1 + ((-41808) + x3 + (20800*x2)), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tl.load(in_ptr2 + (x4), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp9, tmp16, tmp17)
    tmp19 = tl.where(tmp13, tmp18, 0.0)
    tmp20 = tl.load(in_ptr2 + (x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp5, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp12, tmp21)
    tmp23 = tl.where(tmp5, tmp22, 0.0)
    tmp24 = tl.load(in_ptr1 + ((-41808) + x3 + (20800*x2)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp10, tmp24, 0.0)
    tmp26 = tl.where(tmp9, tmp25, tmp20)
    tmp27 = tl.where(tmp5, tmp26, 0.0)
    tmp29 = tl.where(tmp5, tmp27, tmp28)
    tmp30 = tl.where(tmp5, tmp23, tmp29)
    tl.store(out_ptr0 + (x4), tmp30, xmask)
''')
