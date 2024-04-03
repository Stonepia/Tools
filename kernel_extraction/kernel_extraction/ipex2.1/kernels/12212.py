

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/sy/csyil3qepnqyypccnh4kwyih2ptmazpej4x3v7ywypnx7nu2spno.py
# Source Nodes: [add_14, mul_137, mul_138, mul_142, setitem_103, sub_44], Original ATen: [aten.add, aten.copy, aten.mul, aten.slice_scatter, aten.sub]
# add_14 => add_66
# mul_137 => mul_140
# mul_138 => mul_141
# mul_142 => mul_145
# setitem_103 => copy_103, slice_scatter_67, slice_scatter_68
# sub_44 => sub_44
triton_poi_fused_add_copy_mul_slice_scatter_sub_81 = async_compile.triton('triton_poi_fused_add_copy_mul_slice_scatter_sub_81', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_mul_slice_scatter_sub_81', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_mul_slice_scatter_sub_81(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x4 = xindex
    x2 = (xindex // 5304)
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
    tmp10 = 1 + x0
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 27, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp9
    tmp17 = tl.load(in_ptr0 + (31824 + (3*x4)), tmp16 & xmask, other=0.0)
    tmp18 = tl.where(tmp16, tmp17, 0.0)
    tmp19 = tl.full([1], 0.0, tl.float64)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = 2 + x0
    tmp22 = tmp21 >= tmp11
    tmp23 = tmp21 < tmp13
    tmp24 = tmp22 & tmp23
    tmp25 = tmp24 & tmp9
    tmp26 = tl.load(in_ptr1 + (31827 + (3*x4)), tmp25 & xmask, other=0.0)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tl.where(tmp24, tmp27, tmp19)
    tmp29 = tl.load(in_ptr1 + (31824 + (3*x4)), tmp16 & xmask, other=0.0)
    tmp30 = tl.where(tmp16, tmp29, 0.0)
    tmp31 = tl.where(tmp15, tmp30, tmp19)
    tmp32 = tmp28 + tmp31
    tmp33 = tmp20 * tmp32
    tmp34 = tl.full([1], 0.5, tl.float64)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.load(in_ptr2 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp9 & xmask, other=0.0)
    tmp37 = tmp36 * tmp34
    tmp38 = tmp35 - tmp37
    tmp39 = tl.where(tmp9, tmp38, 0.0)
    tmp40 = tl.where(tmp8, tmp39, tmp19)
    tmp41 = tl.where(tmp5, tmp40, 0.0)
    tmp42 = tl.where(tmp5, tmp41, tmp19)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')
