

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/ws/cws7kjejvjaqer5bitvaq63ccgjbklwrwf6ehmeg34ralq6ob4xt.py
# Source Nodes: [mul_100, mul_145, mul_99, neg_54, setitem_104, sub_45, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.copy, aten.div, aten.mul, aten.neg, aten.sub]
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104
# sub_45 => sub_45
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_copy_div_mul_neg_sub_79 = async_compile.triton('triton_poi_fused_copy_div_mul_neg_sub_79', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_neg_sub_79', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_neg_sub_79(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x3 = xindex % 5200
    x4 = xindex
    tmp46 = tl.load(in_ptr3 + (10660 + x3 + (5304*x2)), xmask)
    tmp48 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr5 + (2 + x2), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), xmask)
    tmp53 = tl.load(in_ptr6 + (10634 + x3 + (5304*x2)), xmask)
    tmp55 = tl.load(in_ptr7 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x1
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + (31980 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (47892 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = tl.full([1], 0.5, tl.float64)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr2 + (5200 + x4), tmp11 & xmask, other=0.0)
    tmp20 = tmp19 * tmp17
    tmp21 = tmp18 - tmp20
    tmp22 = tl.where(tmp11, tmp21, 0.0)
    tmp23 = tl.full([1], 0.0, tl.float64)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tl.where(tmp5, tmp24, 0.0)
    tmp26 = tl.where(tmp5, tmp25, tmp23)
    tmp27 = 1 + x2
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp3
    tmp30 = tmp28 & tmp29
    tmp31 = tmp10 & tmp30
    tmp32 = tl.load(in_ptr0 + (16068 + (3*x3) + (15912*x2)), tmp31 & xmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), tmp31 & xmask, other=0.0)
    tmp34 = tl.load(in_ptr1 + (16068 + (3*x3) + (15912*x2)), tmp31 & xmask, other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tmp32 * tmp35
    tmp37 = tmp36 * tmp17
    tmp38 = tl.load(in_ptr2 + (x4), tmp31 & xmask, other=0.0)
    tmp39 = tmp38 * tmp17
    tmp40 = tmp37 - tmp39
    tmp41 = tl.where(tmp31, tmp40, 0.0)
    tmp42 = tl.where(tmp10, tmp41, tmp23)
    tmp43 = tl.where(tmp30, tmp42, 0.0)
    tmp44 = tl.where(tmp30, tmp43, tmp23)
    tmp45 = tmp26 - tmp44
    tmp47 = -tmp45
    tmp50 = tmp48 * tmp49
    tmp51 = tmp47 / tmp50
    tmp54 = tmp52 - tmp53
    tmp56 = tmp48 * tmp55
    tmp57 = tmp54 / tmp56
    tmp58 = tmp51 - tmp57
    tmp59 = tmp46 * tmp58
    tl.store(in_out_ptr0 + (x4), tmp59, xmask)
''')
