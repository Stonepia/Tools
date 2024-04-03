

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/d6/cd6l2voi4obsfhz4cytd3ghh6fneck5kycey7qdhj7jzjtpzkh4b.py
# Source Nodes: [iadd_14], Original ATen: [aten.slice_scatter]
# iadd_14 => slice_scatter_174, slice_scatter_175
triton_poi_fused_slice_scatter_46 = async_compile.triton('triton_poi_fused_slice_scatter_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_46(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x3 = xindex % 5304
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 25, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp9 & xmask, other=0.0)
    tmp11 = tl.where(tmp9, tmp10, 0.0)
    tmp12 = 2 + x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp5
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr1 + ((-52) + x3 + (5200*x2)), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp15 & tmp16
    tmp21 = tl.load(in_ptr2 + (x4), tmp20 & xmask, other=0.0)
    tmp22 = tl.where(tmp20, tmp21, 0.0)
    tmp23 = 0.0
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp5, tmp19, tmp24)
    tmp26 = tl.where(tmp16, tmp25, 0.0)
    tmp27 = tl.load(in_ptr2 + (x4), tmp16 & xmask, other=0.0)
    tmp28 = tl.where(tmp16, tmp27, 0.0)
    tmp29 = tl.where(tmp15, tmp28, tmp23)
    tmp30 = tl.where(tmp15, tmp26, tmp29)
    tmp31 = tl.where(tmp8, tmp11, tmp30)
    tmp32 = tl.where(tmp5, tmp31, 0.0)
    tmp33 = tmp5 & tmp15
    tmp34 = tl.load(in_ptr1 + ((-52) + x3 + (5200*x2)), tmp33 & xmask, other=0.0)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp15 & tmp15
    tmp37 = tl.load(in_ptr2 + (x4), tmp36 & xmask, other=0.0)
    tmp38 = tl.where(tmp36, tmp37, 0.0)
    tmp39 = tl.where(tmp15, tmp38, tmp23)
    tmp40 = tl.where(tmp5, tmp35, tmp39)
    tmp41 = tl.where(tmp15, tmp40, 0.0)
    tmp42 = tl.load(in_ptr2 + (x4), tmp15 & xmask, other=0.0)
    tmp43 = tl.where(tmp15, tmp42, 0.0)
    tmp44 = tl.where(tmp15, tmp43, tmp23)
    tmp45 = tl.where(tmp15, tmp41, tmp44)
    tmp46 = tl.where(tmp5, tmp32, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, xmask)
''')
