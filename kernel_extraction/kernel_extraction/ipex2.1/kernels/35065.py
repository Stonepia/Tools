

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/6c/c6cg3mzhpbd6vlfb336pgbmhctfe6ibeia6d62vtivh7estux6mj.py
# Source Nodes: [iadd_5], Original ATen: [aten.slice_scatter]
# iadd_5 => slice_scatter_96, slice_scatter_97, slice_scatter_98, slice_scatter_99
triton_poi_fused_slice_scatter_29 = async_compile.triton('triton_poi_fused_slice_scatter_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x4 = xindex
    tmp41 = tl.load(in_ptr1 + (10608 + x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp7 & tmp5
    tmp9 = tl.load(in_ptr0 + ((-26) + x0 + (25*x1) + (5025*x2)), tmp8 & xmask, other=0.0)
    tmp10 = tl.where(tmp8, tmp9, 0.0)
    tmp11 = 2 + x2
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp11 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp5
    tmp17 = tmp5 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tl.load(in_ptr1 + (10608 + x4), tmp18 & xmask, other=0.0)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.load(in_ptr1 + (10608 + x4), tmp17 & xmask, other=0.0)
    tmp22 = tl.where(tmp7, tmp20, tmp21)
    tmp23 = tl.where(tmp17, tmp22, 0.0)
    tmp24 = tl.load(in_ptr1 + (10608 + x4), tmp16 & xmask, other=0.0)
    tmp25 = tl.where(tmp5, tmp23, tmp24)
    tmp26 = tl.where(tmp16, tmp25, 0.0)
    tmp27 = tl.load(in_ptr1 + (10608 + x4), tmp5 & xmask, other=0.0)
    tmp28 = tl.where(tmp15, tmp26, tmp27)
    tmp29 = tl.where(tmp7, tmp10, tmp28)
    tmp30 = tl.where(tmp5, tmp29, 0.0)
    tmp31 = tmp5 & tmp15
    tmp32 = tmp7 & tmp31
    tmp33 = tl.load(in_ptr1 + (10608 + x4), tmp32 & xmask, other=0.0)
    tmp34 = tl.where(tmp32, tmp33, 0.0)
    tmp35 = tl.load(in_ptr1 + (10608 + x4), tmp31 & xmask, other=0.0)
    tmp36 = tl.where(tmp7, tmp34, tmp35)
    tmp37 = tl.where(tmp31, tmp36, 0.0)
    tmp38 = tl.load(in_ptr1 + (10608 + x4), tmp15 & xmask, other=0.0)
    tmp39 = tl.where(tmp5, tmp37, tmp38)
    tmp40 = tl.where(tmp15, tmp39, 0.0)
    tmp42 = tl.where(tmp15, tmp40, tmp41)
    tmp43 = tl.where(tmp5, tmp30, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, xmask)
''')
