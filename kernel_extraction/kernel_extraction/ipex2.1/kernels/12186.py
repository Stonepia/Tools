

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/25/c25iwjzbs6rpvznckw5eqlooghidm3qq5mu65tfvtj4ummwukv6k.py
# Source Nodes: [mul_66, mul_67, mul_68, mul_69, setitem_60, setitem_61, setitem_62, sub_2, sub_3, sub_4, sub_5, truediv_38, truediv_39, truediv_40, truediv_41], Original ATen: [aten.copy, aten.div, aten.mul, aten.sub]
# mul_66 => mul_69
# mul_67 => mul_70
# mul_68 => mul_71
# mul_69 => mul_72
# setitem_60 => copy_60
# setitem_61 => copy_61
# setitem_62 => copy_62
# sub_2 => sub_2
# sub_3 => sub_3
# sub_4 => sub_4
# sub_5 => sub_5
# truediv_38 => div_35
# truediv_39 => div_36
# truediv_40 => div_37
# truediv_41 => div_38
triton_poi_fused_copy_div_mul_sub_55 = async_compile.triton('triton_poi_fused_copy_div_mul_sub_55', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: '*fp64', 7: '*fp64', 8: '*fp64', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_sub_55', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_sub_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (23 + (26*x0)), xmask)
    tmp1 = tl.load(in_ptr1 + (23 + (26*x0)), xmask)
    tmp2 = tl.load(in_ptr2 + (24 + (26*x0)), xmask)
    tmp9 = tl.load(in_ptr3 + (x0), xmask)
    tmp10 = tl.load(in_ptr4 + (25 + (26*x0)), xmask)
    tmp12 = tl.load(in_ptr4 + (23 + (26*x0)), xmask)
    tmp16 = tl.load(in_ptr0 + (22 + (26*x0)), xmask)
    tmp17 = tl.load(in_ptr1 + (22 + (26*x0)), xmask)
    tmp19 = tl.load(in_ptr2 + (23 + (26*x0)), xmask)
    tmp25 = tl.load(in_ptr4 + (22 + (26*x0)), xmask)
    tmp29 = tl.load(in_ptr0 + (21 + (26*x0)), xmask)
    tmp30 = tl.load(in_ptr1 + (21 + (26*x0)), xmask)
    tmp33 = tl.load(in_ptr2 + (22 + (26*x0)), xmask)
    tmp40 = tl.load(in_ptr4 + (21 + (26*x0)), xmask)
    tmp44 = tl.load(in_ptr0 + (20 + (26*x0)), xmask)
    tmp45 = tl.load(in_ptr1 + (20 + (26*x0)), xmask)
    tmp49 = tl.load(in_ptr2 + (21 + (26*x0)), xmask)
    tmp57 = tl.load(in_ptr4 + (20 + (26*x0)), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tl.full([1], 23, tl.int32)
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
    tmp23 = tl.full([1], 22, tl.int32)
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
    tmp38 = tl.full([1], 21, tl.int32)
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
    tmp55 = tl.full([1], 20, tl.int32)
    tmp56 = tmp55 == tmp6
    tmp58 = tl.where(tmp56, tmp9, tmp57)
    tmp59 = tl.where(tmp56, tmp11, tmp58)
    tmp60 = tmp54 / tmp59
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
    tl.store(out_ptr2 + (x0), tmp43, xmask)
    tl.store(out_ptr3 + (x0), tmp60, xmask)
''')
