

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zu/czuxgr7ixfmxug25qbdapzjs7febq4owrey4rs6zj3onseo2dmlz.py
# Source Nodes: [setitem_21, setitem_23, setitem_25, setitem_27], Original ATen: [aten.slice_scatter]
# setitem_21 => slice_scatter_95
# setitem_23 => slice_scatter_108
# setitem_25 => slice_scatter_121
# setitem_27 => slice_scatter_134
triton_poi_fused_slice_scatter_52 = async_compile.triton('triton_poi_fused_slice_scatter_52', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_ptr4', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_52', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4328064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21216)
    x2 = xindex
    tmp14 = tl.load(in_ptr4 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-42432) + x2), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + ((-42432) + x2), tmp5 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tl.load(in_ptr2 + ((-42432) + x2), tmp5 & xmask, other=0.0)
    tmp11 = tl.where(tmp5, tmp10, 0.0)
    tmp12 = tl.load(in_ptr3 + ((-42432) + x2), tmp5 & xmask, other=0.0)
    tmp13 = tl.where(tmp5, tmp12, 0.0)
    tmp15 = tl.where(tmp5, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp11, tmp15)
    tmp17 = tl.where(tmp5, tmp9, tmp16)
    tmp18 = tl.where(tmp5, tmp7, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''')
