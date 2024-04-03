

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/xp/cxppujij7czav2yunisafbffk7ky74mvqpl777ulnftreohd6cs4.py
# Source Nodes: [iadd_33, iadd_34, iadd_35, iadd_36, mul_47, mul_48, mul_49, mul_50, neg_35, neg_37, truediv_27, truediv_28], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_33 => add_42
# iadd_34 => add_43
# iadd_35 => add_44
# iadd_36 => add_45
# mul_47 => mul_50
# mul_48 => mul_51
# mul_49 => mul_52
# mul_50 => mul_53
# neg_35 => neg_35
# neg_37 => neg_37
# truediv_27 => div_24
# truediv_28 => div_25
triton_poi_fused_add_div_mul_neg_34 = async_compile.triton('triton_poi_fused_add_div_mul_neg_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp3 = tl.load(in_ptr0 + (16 + (26*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (17 + (26*x2)), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp16 = tl.load(in_ptr2 + (17 + (26*x2)), xmask).to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (16 + (26*x2)), xmask)
    tmp33 = tl.load(in_ptr3 + (17 + (26*x2)), xmask)
    tmp35 = tl.load(in_ptr4 + (16 + (26*x2)), xmask).to(tl.float32)
    tmp43 = tl.load(in_ptr2 + (18 + (26*x2)), xmask).to(tl.float32)
    tmp58 = tl.load(in_ptr3 + (18 + (26*x2)), xmask)
    tmp63 = tl.load(in_ptr4 + (17 + (26*x2)), xmask).to(tl.float32)
    tmp69 = tl.load(in_ptr0 + (18 + (26*x2)), xmask).to(tl.float32)
    tmp0 = tl.full([1], 17, tl.int32)
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 17, tl.int64)
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
    tmp39 = tl.full([1], 18, tl.int64)
    tmp40 = tmp39 >= tmp9
    tmp41 = tmp11 & tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp39 == tmp9
    tmp46 = tmp11 & tmp45
    tmp47 = tmp46 == 0
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp44 * tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp0 == tmp0
    tmp52 = tl.where(tmp51, tmp38, tmp34)
    tmp53 = tl.where(tmp51, tmp52, tmp52)
    tmp54 = tmp50 / tmp53
    tmp55 = tl.full([1], 18, tl.int32)
    tmp56 = tmp55 == tmp0
    tmp57 = tmp55 == tmp1
    tmp59 = tl.where(tmp57, tmp25, tmp58)
    tmp60 = tl.where(tmp56, tmp38, tmp59)
    tmp61 = tl.where(tmp56, tmp52, tmp60)
    tmp62 = -tmp54
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp62 * tmp64
    tmp66 = tmp61 + tmp65
    tmp67 = tmp32.to(tl.float32)
    tmp68 = tl.where(tmp51, tmp67, tmp5)
    tmp70 = tl.where(tmp57, tmp3, tmp69)
    tmp71 = tl.where(tmp56, tmp67, tmp70)
    tmp72 = tl.where(tmp56, tmp68, tmp71)
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tl.where(tmp51, tmp68, tmp68)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp62 * tmp75
    tmp77 = tmp73 + tmp76
    tl.store(out_ptr0 + (x2), tmp32, xmask)
    tl.store(out_ptr1 + (x2), tmp38, xmask)
    tl.store(out_ptr3 + (x2), tmp66, xmask)
    tl.store(out_ptr4 + (x2), tmp77, xmask)
''')
