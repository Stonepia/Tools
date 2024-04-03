

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/5l/c5lr5pxcu5u6x32c4vwzfqfwv43okgtgdpbsodqayck2hmbbd2hx.py
# Source Nodes: [iadd_41, iadd_43, iadd_44, mul_55, mul_57, mul_58, neg_43, neg_45, truediv_31, truediv_32], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_41 => add_50
# iadd_43 => add_52
# iadd_44 => add_53
# mul_55 => mul_58
# mul_57 => mul_60
# mul_58 => mul_61
# neg_43 => neg_43
# neg_45 => neg_45
# truediv_31 => div_28
# truediv_32 => div_29
triton_poi_fused_add_div_mul_neg_44 = async_compile.triton('triton_poi_fused_add_div_mul_neg_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp0 = tl.load(in_ptr0 + (21 + (26*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp10 = tl.load(in_ptr2 + (21 + (26*x2)), xmask)
    tmp17 = tl.load(in_ptr0 + (20 + (26*x2)), xmask)
    tmp20 = tl.load(in_ptr3 + (20 + (26*x2)), xmask)
    tmp23 = tl.load(in_ptr4 + (22 + (26*x2)), xmask)
    tmp28 = tl.load(in_ptr2 + (22 + (26*x2)), xmask)
    tmp41 = tl.load(in_ptr4 + (21 + (26*x2)), xmask)
    tmp46 = tl.load(in_ptr0 + (22 + (26*x2)), xmask)
    tmp49 = tl.load(in_ptr3 + (21 + (26*x2)), xmask)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 21, tl.int64)
    tmp7 = tmp6 >= tmp3
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 == tmp3
    tmp13 = tmp5 & tmp12
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp11 * tmp15
    tmp18 = tmp16 / tmp17
    tmp19 = -tmp18
    tmp21 = tmp19 * tmp20
    tmp22 = tmp0 + tmp21
    tmp24 = tl.full([1], 22, tl.int64)
    tmp25 = tmp24 >= tmp3
    tmp26 = tmp5 & tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp24 == tmp3
    tmp31 = tmp5 & tmp30
    tmp32 = tmp31 == 0
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp29 * tmp33
    tmp35 = tl.full([1], 21, tl.int32)
    tmp36 = tmp35 == tmp35
    tmp37 = tl.where(tmp36, tmp22, tmp0)
    tmp38 = tl.where(tmp36, tmp37, tmp37)
    tmp39 = tmp34 / tmp38
    tmp40 = -tmp39
    tmp42 = tmp40 * tmp41
    tmp43 = tmp23 + tmp42
    tmp44 = tl.full([1], 22, tl.int32)
    tmp45 = tmp44 == tmp35
    tmp47 = tl.where(tmp45, tmp22, tmp46)
    tmp48 = tl.where(tmp45, tmp37, tmp47)
    tmp50 = tmp40 * tmp49
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x2), tmp22, xmask)
    tl.store(out_ptr1 + (x2), tmp43, xmask)
    tl.store(out_ptr2 + (x2), tmp51, xmask)
''')
