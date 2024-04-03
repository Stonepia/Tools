

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/rj/crjm5lhc626rgkb4cnr4363rzfpi6egkuxvjruybnb6drgp3n6en.py
# Source Nodes: [mul_88, mul_89, sub_24, sub_25, truediv_60, truediv_61], Original ATen: [aten.div, aten.mul, aten.sub]
# mul_88 => mul_91
# mul_89 => mul_92
# sub_24 => sub_24
# sub_25 => sub_25
# truediv_60 => div_57
# truediv_61 => div_58
triton_poi_fused_div_mul_sub_58 = async_compile.triton('triton_poi_fused_div_mul_sub_58', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_sub_58', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_div_mul_sub_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr0 + (x0), xmask)
    tmp8 = tl.load(in_ptr1 + (24 + (26*x0)), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (25 + (26*x0)), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (1 + (26*x0)), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (1 + (26*x0)), xmask).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (2 + (26*x0)), xmask).to(tl.float32)
    tmp22 = tl.load(in_ptr4 + (x0), xmask)
    tmp23 = tl.load(in_ptr5 + (24 + (26*x0)), xmask)
    tmp24 = tl.load(in_ptr5 + (25 + (26*x0)), xmask)
    tmp27 = tl.load(in_ptr5 + (1 + (26*x0)), xmask)
    tmp35 = tl.load(in_ptr1 + (26*x0), xmask).to(tl.float32)
    tmp39 = tl.load(in_ptr2 + (26*x0), xmask).to(tl.float32)
    tmp42 = tl.load(in_ptr3 + (1 + (26*x0)), xmask).to(tl.float32)
    tmp47 = tl.load(in_ptr5 + (26*x0), xmask)
    tmp0 = tl.full([1], 1, tl.int32)
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
    tmp32 = tl.full([1], 0, tl.int32)
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
    tl.store(out_ptr0 + (x0), tmp31, xmask)
    tl.store(out_ptr1 + (x0), tmp51, xmask)
''')
