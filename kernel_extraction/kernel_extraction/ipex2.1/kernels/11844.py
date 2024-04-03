

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/jb/cjbqnayrur5npesvpxh7lx7ywrato4ia6f3pflu6uygtwkegit6o.py
# Source Nodes: [mul_82, mul_83, mul_84, mul_85, setitem_76, setitem_77, setitem_78, sub_18, sub_19, sub_20, sub_21, truediv_54, truediv_55, truediv_56, truediv_57], Original ATen: [aten.copy, aten.div, aten.mul, aten.sub]
# mul_82 => mul_85
# mul_83 => mul_86
# mul_84 => mul_87
# mul_85 => mul_88
# setitem_76 => copy_76
# setitem_77 => copy_77
# setitem_78 => copy_78
# sub_18 => sub_18
# sub_19 => sub_19
# sub_20 => sub_20
# sub_21 => sub_21
# truediv_54 => div_51
# truediv_55 => div_52
# truediv_56 => div_53
# truediv_57 => div_54
triton_poi_fused_copy_div_mul_sub_62 = async_compile.triton('triton_poi_fused_copy_div_mul_sub_62', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_sub_62', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_sub_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + (26*x0)), xmask)
    tmp1 = tl.load(in_ptr1 + (7 + (26*x0)), xmask)
    tmp2 = tl.load(in_ptr2 + (8 + (26*x0)), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tl.load(in_ptr4 + (25 + (26*x0)), xmask)
    tmp12 = tl.load(in_ptr4 + (7 + (26*x0)), xmask)
    tmp16 = tl.load(in_ptr0 + (6 + (26*x0)), xmask)
    tmp17 = tl.load(in_ptr1 + (6 + (26*x0)), xmask)
    tmp19 = tl.load(in_ptr2 + (7 + (26*x0)), xmask)
    tmp25 = tl.load(in_ptr4 + (6 + (26*x0)), xmask)
    tmp29 = tl.load(in_ptr0 + (5 + (26*x0)), xmask)
    tmp30 = tl.load(in_ptr1 + (5 + (26*x0)), xmask)
    tmp33 = tl.load(in_ptr2 + (6 + (26*x0)), xmask)
    tmp40 = tl.load(in_ptr4 + (5 + (26*x0)), xmask)
    tmp44 = tl.load(in_ptr0 + (4 + (26*x0)), xmask)
    tmp45 = tl.load(in_ptr1 + (4 + (26*x0)), xmask)
    tmp49 = tl.load(in_ptr2 + (5 + (26*x0)), xmask)
    tmp57 = tl.load(in_ptr4 + (4 + (26*x0)), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tl.full([1], 7, tl.int32)
    tmp6 = tl.full([1], 25, tl.int32)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp6 == tmp6
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp13 = tl.where(tmp7, tmp9, tmp12)
    tmp14 = tl.where(tmp7, tmp11, tmp13)
    tmp15 = tmp4 / tmp14
    tmp18 = tmp5 == tmp5
    tmp20 = tl.where(tmp18, tmp15, tmp19)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp16 - tmp21
    tmp23 = tl.full([1], 6, tl.int32)
    tmp24 = tmp23 == tmp6
    tmp26 = tl.where(tmp24, tmp9, tmp25)
    tmp27 = tl.where(tmp24, tmp11, tmp26)
    tmp28 = tmp22 / tmp27
    tmp31 = tmp23 == tmp23
    tmp32 = tmp23 == tmp5
    tmp34 = tl.where(tmp32, tmp15, tmp33)
    tmp35 = tl.where(tmp31, tmp28, tmp34)
    tmp36 = tmp30 * tmp35
    tmp37 = tmp29 - tmp36
    tmp38 = tl.full([1], 5, tl.int32)
    tmp39 = tmp38 == tmp6
    tmp41 = tl.where(tmp39, tmp9, tmp40)
    tmp42 = tl.where(tmp39, tmp11, tmp41)
    tmp43 = tmp37 / tmp42
    tmp46 = tmp38 == tmp38
    tmp47 = tmp38 == tmp23
    tmp48 = tmp38 == tmp5
    tmp50 = tl.where(tmp48, tmp15, tmp49)
    tmp51 = tl.where(tmp47, tmp28, tmp50)
    tmp52 = tl.where(tmp46, tmp43, tmp51)
    tmp53 = tmp45 * tmp52
    tmp54 = tmp44 - tmp53
    tmp55 = tl.full([1], 4, tl.int32)
    tmp56 = tmp55 == tmp6
    tmp58 = tl.where(tmp56, tmp9, tmp57)
    tmp59 = tl.where(tmp56, tmp11, tmp58)
    tmp60 = tmp54 / tmp59
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp43, xmask)
    tl.store(out_ptr3 + (x0), tmp60, xmask)
''')
