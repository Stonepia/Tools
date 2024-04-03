

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/e2/ce2x27lv35bkw5zes7xejgp4ff5er64gcfjoetd2w7orifgbf3uu.py
# Source Nodes: [mul_80, mul_81, mul_82, mul_83, setitem_77, sub_16, sub_17, sub_18, sub_19, truediv_52, truediv_53, truediv_54, truediv_55], Original ATen: [aten.copy, aten.div, aten.mul, aten.sub]
# mul_80 => mul_83
# mul_81 => mul_84
# mul_82 => mul_85
# mul_83 => mul_86
# setitem_77 => copy_77
# sub_16 => sub_16
# sub_17 => sub_17
# sub_18 => sub_18
# sub_19 => sub_19
# truediv_52 => div_49
# truediv_53 => div_50
# truediv_54 => div_51
# truediv_55 => div_52
triton_poi_fused_copy_div_mul_sub_54 = async_compile.triton('triton_poi_fused_copy_div_mul_sub_54', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_sub_54', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_sub_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (24 + (26*x0)), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (25 + (26*x0)), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (9 + (26*x0)), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (9 + (26*x0)), xmask).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (10 + (26*x0)), xmask).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (x0), xmask)
    tmp23 = tl.load(in_ptr5 + (24 + (26*x0)), xmask)
    tmp24 = tl.load(in_ptr5 + (25 + (26*x0)), xmask)
    tmp27 = tl.load(in_ptr5 + (9 + (26*x0)), xmask)
    tmp35 = tl.load(in_ptr1 + (8 + (26*x0)), xmask).to(tl.float32)
    tmp39 = tl.load(in_ptr2 + (8 + (26*x0)), xmask).to(tl.float32)
    tmp42 = tl.load(in_ptr3 + (9 + (26*x0)), xmask).to(tl.float32)
    tmp47 = tl.load(in_ptr5 + (8 + (26*x0)), xmask)
    tmp55 = tl.load(in_ptr1 + (7 + (26*x0)), xmask).to(tl.float32)
    tmp59 = tl.load(in_ptr2 + (7 + (26*x0)), xmask).to(tl.float32)
    tmp63 = tl.load(in_ptr3 + (8 + (26*x0)), xmask).to(tl.float32)
    tmp69 = tl.load(in_ptr5 + (7 + (26*x0)), xmask)
    tmp77 = tl.load(in_ptr1 + (6 + (26*x0)), xmask).to(tl.float32)
    tmp81 = tl.load(in_ptr2 + (6 + (26*x0)), xmask).to(tl.float32)
    tmp86 = tl.load(in_ptr3 + (7 + (26*x0)), xmask).to(tl.float32)
    tmp93 = tl.load(in_ptr5 + (6 + (26*x0)), xmask)
    tmp0 = tl.full([1], 9, tl.int32)
    tmp1 = tl.full([1], 25, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full([1], 24, tl.int32)
    tmp7 = tmp1 == tmp6
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.where(tmp3, tmp5, tmp10)
    tmp12 = tmp0 == tmp6
    tmp14 = tl.where(tmp12, tmp8, tmp13)
    tmp15 = tl.where(tmp2, tmp5, tmp14)
    tmp16 = tl.where(tmp2, tmp11, tmp15)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp25 = tl.where(tmp7, tmp23, tmp24)
    tmp26 = tl.where(tmp3, tmp22, tmp25)
    tmp28 = tl.where(tmp12, tmp23, tmp27)
    tmp29 = tl.where(tmp2, tmp22, tmp28)
    tmp30 = tl.where(tmp2, tmp26, tmp29)
    tmp31 = tmp21 / tmp30
    tmp32 = tl.full([1], 8, tl.int32)
    tmp33 = tmp32 == tmp1
    tmp34 = tmp32 == tmp6
    tmp36 = tl.where(tmp34, tmp8, tmp35)
    tmp37 = tl.where(tmp33, tmp5, tmp36)
    tmp38 = tl.where(tmp33, tmp11, tmp37)
    tmp40 = tmp0 == tmp0
    tmp41 = tmp31.to(tl.float32)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tmp39 * tmp43
    tmp45 = tmp38 - tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp48 = tl.where(tmp34, tmp23, tmp47)
    tmp49 = tl.where(tmp33, tmp22, tmp48)
    tmp50 = tl.where(tmp33, tmp26, tmp49)
    tmp51 = tmp46 / tmp50
    tmp52 = tl.full([1], 7, tl.int32)
    tmp53 = tmp52 == tmp1
    tmp54 = tmp52 == tmp6
    tmp56 = tl.where(tmp54, tmp8, tmp55)
    tmp57 = tl.where(tmp53, tmp5, tmp56)
    tmp58 = tl.where(tmp53, tmp11, tmp57)
    tmp60 = tmp32 == tmp32
    tmp61 = tmp51.to(tl.float32)
    tmp62 = tmp32 == tmp0
    tmp64 = tl.where(tmp62, tmp41, tmp63)
    tmp65 = tl.where(tmp60, tmp61, tmp64)
    tmp66 = tmp59 * tmp65
    tmp67 = tmp58 - tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp70 = tl.where(tmp54, tmp23, tmp69)
    tmp71 = tl.where(tmp53, tmp22, tmp70)
    tmp72 = tl.where(tmp53, tmp26, tmp71)
    tmp73 = tmp68 / tmp72
    tmp74 = tl.full([1], 6, tl.int32)
    tmp75 = tmp74 == tmp1
    tmp76 = tmp74 == tmp6
    tmp78 = tl.where(tmp76, tmp8, tmp77)
    tmp79 = tl.where(tmp75, tmp5, tmp78)
    tmp80 = tl.where(tmp75, tmp11, tmp79)
    tmp82 = tmp52 == tmp52
    tmp83 = tmp73.to(tl.float32)
    tmp84 = tmp52 == tmp32
    tmp85 = tmp52 == tmp0
    tmp87 = tl.where(tmp85, tmp41, tmp86)
    tmp88 = tl.where(tmp84, tmp61, tmp87)
    tmp89 = tl.where(tmp82, tmp83, tmp88)
    tmp90 = tmp81 * tmp89
    tmp91 = tmp80 - tmp90
    tmp92 = tmp91.to(tl.float32)
    tmp94 = tl.where(tmp76, tmp23, tmp93)
    tmp95 = tl.where(tmp75, tmp22, tmp94)
    tmp96 = tl.where(tmp75, tmp26, tmp95)
    tmp97 = tmp92 / tmp96
    tmp98 = tmp97.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp31, xmask)
    tl.store(out_ptr1 + (x0), tmp51, xmask)
    tl.store(out_ptr2 + (x0), tmp73, xmask)
    tl.store(in_out_ptr0 + (x0), tmp98, xmask)
''')
