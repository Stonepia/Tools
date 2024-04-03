

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/j5/cj5bmmslsboykh6ull5lpo6lu6zabvsdflkjip2nsawyz5ehrtue.py
# Source Nodes: [iadd_29, iadd_30, iadd_31, mul_43, mul_44, mul_45, neg_31, neg_33, truediv_25, truediv_26], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.neg]
# iadd_29 => add_38
# iadd_30 => add_39, convert_element_type_14
# iadd_31 => add_40
# mul_43 => mul_46
# mul_44 => mul_47
# mul_45 => mul_48
# neg_31 => neg_31
# neg_33 => neg_33
# truediv_25 => div_22
# truediv_26 => div_23
triton_poi_fused__to_copy_add_div_mul_neg_31 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_neg_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_neg_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_neg_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp0 = tl.load(in_ptr0 + (15 + (26*x2)), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp11 = tl.load(in_ptr2 + (15 + (26*x2)), xmask).to(tl.float32)
    tmp21 = tl.load(in_ptr3 + (14 + (26*x2)), xmask)
    tmp25 = tl.load(in_ptr0 + (14 + (26*x2)), xmask).to(tl.float32)
    tmp32 = tl.load(in_ptr3 + (15 + (26*x2)), xmask)
    tmp34 = tl.load(in_ptr4 + (14 + (26*x2)), xmask).to(tl.float32)
    tmp42 = tl.load(in_ptr2 + (16 + (26*x2)), xmask).to(tl.float32)
    tmp57 = tl.load(in_ptr3 + (16 + (26*x2)), xmask)
    tmp62 = tl.load(in_ptr4 + (15 + (26*x2)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp2 - tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 15, tl.int64)
    tmp8 = tmp7 >= tmp4
    tmp9 = tmp6 & tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 == tmp4
    tmp14 = tmp6 & tmp13
    tmp15 = tmp14 == 0
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp12 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1], 14, tl.int32)
    tmp20 = tmp19 == tmp19
    tmp22 = tl.where(tmp20, tmp21, tmp21)
    tmp23 = tmp18 / tmp22
    tmp24 = -tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp1 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.full([1], 15, tl.int32)
    tmp31 = tmp30 == tmp19
    tmp33 = tl.where(tmp31, tmp21, tmp32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp24 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full([1], 16, tl.int64)
    tmp39 = tmp38 >= tmp4
    tmp40 = tmp6 & tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp43 = tmp41 * tmp42
    tmp44 = tmp38 == tmp4
    tmp45 = tmp6 & tmp44
    tmp46 = tmp45 == 0
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp43 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp30 == tmp30
    tmp51 = tl.where(tmp50, tmp37, tmp33)
    tmp52 = tl.where(tmp50, tmp51, tmp51)
    tmp53 = tmp49 / tmp52
    tmp54 = tl.full([1], 16, tl.int32)
    tmp55 = tmp54 == tmp30
    tmp56 = tmp54 == tmp19
    tmp58 = tl.where(tmp56, tmp21, tmp57)
    tmp59 = tl.where(tmp55, tmp37, tmp58)
    tmp60 = tl.where(tmp55, tmp51, tmp59)
    tmp61 = -tmp53
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp61 * tmp63
    tmp65 = tmp60 + tmp64
    tl.store(out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr1 + (x2), tmp37, xmask)
    tl.store(out_ptr2 + (x2), tmp53, xmask)
    tl.store(out_ptr3 + (x2), tmp65, xmask)
''')
