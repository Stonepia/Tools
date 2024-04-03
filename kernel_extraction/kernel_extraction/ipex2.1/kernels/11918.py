

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/mv/cmvkypr3yfhsbdgrxlldfn2gmwfsbfcjvp2srbjhh3s5il5u5jj2.py
# Source Nodes: [iadd_49, iadd_50, mul_63, mul_64, mul_65, mul_66, mul_67, neg_51, setitem_61, sub_1, sub_2, sub_3, truediv_35, truediv_36, truediv_37, truediv_38, truediv_39], Original ATen: [aten.add, aten.copy, aten.div, aten.mul, aten.neg, aten.sub]
# iadd_49 => add_58
# iadd_50 => add_59
# mul_63 => mul_66
# mul_64 => mul_67
# mul_65 => mul_68
# mul_66 => mul_69
# mul_67 => mul_70
# neg_51 => neg_51
# setitem_61 => copy_61
# sub_1 => sub_1
# sub_2 => sub_2
# sub_3 => sub_3
# truediv_35 => div_32
# truediv_36 => div_33
# truediv_37 => div_34
# truediv_38 => div_35
# truediv_39 => div_36
triton_poi_fused_add_copy_div_mul_neg_sub_46 = async_compile.triton('triton_poi_fused_add_copy_div_mul_neg_sub_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i64', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy_div_mul_neg_sub_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_copy_div_mul_neg_sub_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp3 = tl.load(in_ptr0 + (24 + (26*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (25 + (26*x2)), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp16 = tl.load(in_ptr2 + (25 + (26*x2)), xmask).to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (24 + (26*x2)), xmask)
    tmp33 = tl.load(in_ptr3 + (25 + (26*x2)), xmask)
    tmp35 = tl.load(in_ptr4 + (24 + (26*x2)), xmask).to(tl.float32)
    tmp51 = tl.load(in_ptr5 + (25 + (26*x2)), xmask).to(tl.float32)
    tmp62 = tl.load(in_ptr0 + (23 + (26*x2)), xmask).to(tl.float32)
    tmp66 = tl.load(in_ptr4 + (23 + (26*x2)), xmask).to(tl.float32)
    tmp68 = tl.load(in_ptr5 + (24 + (26*x2)), xmask).to(tl.float32)
    tmp74 = tl.load(in_ptr3 + (23 + (26*x2)), xmask)
    tmp82 = tl.load(in_ptr0 + (22 + (26*x2)), xmask).to(tl.float32)
    tmp86 = tl.load(in_ptr4 + (22 + (26*x2)), xmask).to(tl.float32)
    tmp89 = tl.load(in_ptr5 + (23 + (26*x2)), xmask).to(tl.float32)
    tmp96 = tl.load(in_ptr3 + (22 + (26*x2)), xmask)
    tmp0 = tl.full([1], 25, tl.int32)
    tmp1 = tl.full([1], 24, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 25, tl.int64)
    tmp13 = tmp12 >= tmp9
    tmp14 = tmp11 & tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp12 == tmp9
    tmp19 = tmp11 & tmp18
    tmp20 = tmp19 == 0
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp1 == tmp1
    tmp26 = tl.where(tmp24, tmp25, tmp25)
    tmp27 = tmp23 / tmp26
    tmp28 = -tmp27
    tmp29 = tl.where(tmp24, tmp3, tmp3)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp32 = tmp6 + tmp31
    tmp34 = tl.where(tmp2, tmp25, tmp33)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp28 * tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = tmp0 == tmp0
    tmp40 = tmp32.to(tl.float32)
    tmp41 = tl.where(tmp39, tmp40, tmp5)
    tmp42 = tl.where(tmp39, tmp41, tmp41)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp39, tmp38, tmp34)
    tmp45 = tl.where(tmp39, tmp44, tmp44)
    tmp46 = tmp43 / tmp45
    tmp47 = tmp1 == tmp0
    tmp48 = tl.where(tmp47, tmp40, tmp29)
    tmp49 = tl.where(tmp47, tmp41, tmp48)
    tmp50 = tmp46.to(tl.float32)
    tmp52 = tl.where(tmp39, tmp50, tmp51)
    tmp53 = tmp35 * tmp52
    tmp54 = tmp49 - tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tl.where(tmp47, tmp38, tmp26)
    tmp57 = tl.where(tmp47, tmp44, tmp56)
    tmp58 = tmp55 / tmp57
    tmp59 = tl.full([1], 23, tl.int32)
    tmp60 = tmp59 == tmp0
    tmp61 = tmp59 == tmp1
    tmp63 = tl.where(tmp61, tmp3, tmp62)
    tmp64 = tl.where(tmp60, tmp40, tmp63)
    tmp65 = tl.where(tmp60, tmp41, tmp64)
    tmp67 = tmp58.to(tl.float32)
    tmp69 = tl.where(tmp47, tmp50, tmp68)
    tmp70 = tl.where(tmp24, tmp67, tmp69)
    tmp71 = tmp66 * tmp70
    tmp72 = tmp65 - tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp75 = tl.where(tmp61, tmp25, tmp74)
    tmp76 = tl.where(tmp60, tmp38, tmp75)
    tmp77 = tl.where(tmp60, tmp44, tmp76)
    tmp78 = tmp73 / tmp77
    tmp79 = tl.full([1], 22, tl.int32)
    tmp80 = tmp79 == tmp0
    tmp81 = tmp79 == tmp1
    tmp83 = tl.where(tmp81, tmp3, tmp82)
    tmp84 = tl.where(tmp80, tmp40, tmp83)
    tmp85 = tl.where(tmp80, tmp41, tmp84)
    tmp87 = tmp59 == tmp59
    tmp88 = tmp78.to(tl.float32)
    tmp90 = tl.where(tmp60, tmp50, tmp89)
    tmp91 = tl.where(tmp61, tmp67, tmp90)
    tmp92 = tl.where(tmp87, tmp88, tmp91)
    tmp93 = tmp86 * tmp92
    tmp94 = tmp85 - tmp93
    tmp95 = tmp94.to(tl.float32)
    tmp97 = tl.where(tmp81, tmp25, tmp96)
    tmp98 = tl.where(tmp80, tmp38, tmp97)
    tmp99 = tl.where(tmp80, tmp44, tmp98)
    tmp100 = tmp95 / tmp99
    tmp101 = tmp100.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp32, xmask)
    tl.store(out_ptr1 + (x2), tmp38, xmask)
    tl.store(out_ptr2 + (x2), tmp46, xmask)
    tl.store(out_ptr3 + (x2), tmp58, xmask)
    tl.store(out_ptr4 + (x2), tmp78, xmask)
    tl.store(in_out_ptr0 + (x2), tmp101, xmask)
''')
