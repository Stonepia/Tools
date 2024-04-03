

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/kw/ckwufqykuteptrh4ck3vhbs326itnstu2u5hyoy2ypfw7fhrcggs.py
# Source Nodes: [iadd_25, iadd_26, iadd_27, mul_39, mul_40, mul_41, neg_27, neg_29, truediv_23, truediv_24], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_25 => add_34
# iadd_26 => add_35
# iadd_27 => add_36
# mul_39 => mul_42
# mul_40 => mul_43
# mul_41 => mul_44
# neg_27 => neg_27
# neg_29 => neg_29
# truediv_23 => div_20
# truediv_24 => div_21
triton_poi_fused_add_div_mul_neg_27 = async_compile.triton('triton_poi_fused_add_div_mul_neg_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200
    x1 = (xindex // 200)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (410 + x0 + (204*x1)), xmask)
    tmp9 = tl.load(in_ptr1 + (13 + (26*x2)), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (12 + (26*x2)), xmask)
    tmp25 = tl.load(in_ptr3 + (11 + (26*x2)), xmask).to(tl.float32)
    tmp26 = tl.load(in_ptr3 + (12 + (26*x2)), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr4 + (x2), xmask)
    tmp38 = tl.load(in_ptr2 + (13 + (26*x2)), xmask)
    tmp40 = tl.load(in_ptr5 + (12 + (26*x2)), xmask).to(tl.float32)
    tmp48 = tl.load(in_ptr1 + (14 + (26*x2)), xmask).to(tl.float32)
    tmp63 = tl.load(in_ptr2 + (14 + (26*x2)), xmask)
    tmp68 = tl.load(in_ptr5 + (13 + (26*x2)), xmask).to(tl.float32)
    tmp73 = tl.load(in_ptr3 + (13 + (26*x2)), xmask).to(tl.float32)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = tl.full([1], 13, tl.int64)
    tmp6 = tmp5 >= tmp2
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 == tmp2
    tmp12 = tmp4 & tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full([1], 12, tl.int32)
    tmp18 = tmp17 == tmp17
    tmp20 = tl.where(tmp18, tmp19, tmp19)
    tmp21 = tmp16 / tmp20
    tmp22 = -tmp21
    tmp23 = tl.full([1], 11, tl.int32)
    tmp24 = tmp17 == tmp23
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.where(tmp18, tmp31, tmp27)
    tmp33 = tl.where(tmp18, tmp32, tmp32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp22 * tmp34
    tmp36 = tl.full([1], 13, tl.int32)
    tmp37 = tmp36 == tmp17
    tmp39 = tl.where(tmp37, tmp19, tmp38)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp22 * tmp41
    tmp43 = tmp39 + tmp42
    tmp44 = tl.full([1], 14, tl.int64)
    tmp45 = tmp44 >= tmp2
    tmp46 = tmp4 & tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp49 = tmp47 * tmp48
    tmp50 = tmp44 == tmp2
    tmp51 = tmp4 & tmp50
    tmp52 = tmp51 == 0
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp49 * tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp36 == tmp36
    tmp57 = tl.where(tmp56, tmp43, tmp39)
    tmp58 = tl.where(tmp56, tmp57, tmp57)
    tmp59 = tmp55 / tmp58
    tmp60 = tl.full([1], 14, tl.int32)
    tmp61 = tmp60 == tmp36
    tmp62 = tmp60 == tmp17
    tmp64 = tl.where(tmp62, tmp19, tmp63)
    tmp65 = tl.where(tmp61, tmp43, tmp64)
    tmp66 = tl.where(tmp61, tmp57, tmp65)
    tmp67 = -tmp59
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp67 * tmp69
    tmp71 = tmp66 + tmp70
    tmp72 = tmp36 == tmp23
    tmp74 = tl.where(tmp72, tmp25, tmp73)
    tmp75 = tl.where(tmp37, tmp31, tmp74)
    tmp76 = tl.where(tmp37, tmp32, tmp75)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tmp77 + tmp35
    tl.store(out_ptr0 + (x2), tmp43, xmask)
    tl.store(out_ptr1 + (x2), tmp59, xmask)
    tl.store(out_ptr2 + (x2), tmp71, xmask)
    tl.store(in_out_ptr0 + (x2), tmp78, xmask)
''')
