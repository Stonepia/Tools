

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/sp/cspjwrs3qpmnftqb75nu4qdfcc7ldptn66te7zp7woklduwgxtog.py
# Source Nodes: [iadd_15], Original ATen: [aten.slice_scatter]
# iadd_15 => slice_scatter_180
triton_poi_fused_slice_scatter_48 = async_compile.triton('triton_poi_fused_slice_scatter_48', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_48', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5200)
    x4 = xindex % 5200
    x1 = (xindex // 26) % 200
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (25*x3)), tmp2 & xmask, other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = 2 + x2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 202, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tl.load(in_ptr1 + (52 + x4 + (5304*x2)), tmp10 & xmask, other=0.0)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 2 + x1
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + (x5), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp10 & tmp10
    tmp21 = tl.load(in_ptr3 + (52 + x4 + (5304*x2)), tmp20 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = tl.full([1], 0.0, tl.float64)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tl.where(tmp16, tmp19, tmp24)
    tmp26 = tl.where(tmp10, tmp25, 0.0)
    tmp27 = tl.load(in_ptr3 + (52 + x4 + (5304*x2)), tmp10 & xmask, other=0.0)
    tmp28 = tl.where(tmp10, tmp27, 0.0)
    tmp29 = tl.where(tmp10, tmp28, tmp23)
    tmp30 = tl.where(tmp10, tmp26, tmp29)
    tmp31 = tl.where(tmp10, tmp12, tmp30)
    tmp32 = tl.where(tmp2, tmp4, tmp31)
    tl.store(out_ptr0 + (x5), tmp32, xmask)
''')
