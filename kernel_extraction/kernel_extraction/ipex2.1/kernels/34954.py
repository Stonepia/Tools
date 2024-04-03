

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/fd/cfdw4k77b55sdgj65yzsifnptzwpdks23xfdubreewlf5tiyfixr.py
# Source Nodes: [setitem_21, setitem_23], Original ATen: [aten.slice_scatter]
# setitem_21 => slice_scatter_94, slice_scatter_95
# setitem_23 => slice_scatter_107, slice_scatter_108
triton_poi_fused_slice_scatter_24 = async_compile.triton('triton_poi_fused_slice_scatter_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 21216)
    x1 = (xindex // 104) % 204
    x3 = xindex % 21216
    x4 = xindex
    tmp29 = tl.load(in_ptr2 + (x4), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + ((-41912) + x3 + (20904*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tmp5 & tmp5
    tmp15 = tmp10 & tmp14
    tmp16 = tl.load(in_ptr1 + ((-41912) + x3 + (20904*x2)), tmp15 & xmask, other=0.0)
    tmp17 = tl.where(tmp15, tmp16, 0.0)
    tmp18 = tl.load(in_ptr2 + (x4), tmp14 & xmask, other=0.0)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tmp20 = tl.where(tmp14, tmp19, 0.0)
    tmp21 = tl.load(in_ptr2 + (x4), tmp5 & xmask, other=0.0)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp13, tmp22)
    tmp24 = tl.where(tmp5, tmp23, 0.0)
    tmp25 = tl.load(in_ptr1 + ((-41912) + x3 + (20904*x2)), tmp11 & xmask, other=0.0)
    tmp26 = tl.where(tmp11, tmp25, 0.0)
    tmp27 = tl.where(tmp10, tmp26, tmp21)
    tmp28 = tl.where(tmp5, tmp27, 0.0)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.where(tmp5, tmp24, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''')
