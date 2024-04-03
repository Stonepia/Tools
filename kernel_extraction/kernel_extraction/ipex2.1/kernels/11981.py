

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/g5/cg5kb7rjvc2pyofaeia52hshlq6gdqe2nqsmjnzvvenoc574i4q4.py
# Source Nodes: [iadd_21, mul_35, mul_37, mul_38, neg_23, neg_25, truediv_21, truediv_22], Original ATen: [aten.add, aten.div, aten.mul, aten.neg]
# iadd_21 => add_30
# mul_35 => mul_38
# mul_37 => mul_40
# mul_38 => mul_41
# neg_23 => neg_23
# neg_25 => neg_25
# truediv_21 => div_18
# truediv_22 => div_19
triton_poi_fused_add_div_mul_neg_25 = async_compile.triton('triton_poi_fused_add_div_mul_neg_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_mul_neg_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp0 = tl.load(in_ptr0 + (11 + (26*x2)), xmask)
    tmp1 = tl.load(in_ptr1 + (410 + x0 + (204*x1)), xmask)
    tmp10 = tl.load(in_ptr2 + (11 + (26*x2)), xmask).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (10 + (26*x2)), xmask)
    tmp21 = tl.load(in_ptr3 + (10 + (26*x2)), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr2 + (12 + (26*x2)), xmask).to(tl.float32)
    tmp43 = tl.load(in_ptr4 + (11 + (26*x2)), xmask).to(tl.float32)
    tmp47 = tl.load(in_ptr3 + (11 + (26*x2)), xmask).to(tl.float32)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 11, tl.int64)
    tmp7 = tmp6 >= tmp3
    tmp8 = tmp5 & tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 == tmp3
    tmp13 = tmp5 & tmp12
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp11 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 / tmp18
    tmp20 = -tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp24 = tmp0 + tmp23
    tmp25 = tl.full([1], 12, tl.int64)
    tmp26 = tmp25 >= tmp3
    tmp27 = tmp5 & tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp25 == tmp3
    tmp32 = tmp5 & tmp31
    tmp33 = tmp32 == 0
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp30 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tl.full([1], 11, tl.int32)
    tmp38 = tmp37 == tmp37
    tmp39 = tl.where(tmp38, tmp24, tmp0)
    tmp40 = tl.where(tmp38, tmp39, tmp39)
    tmp41 = tmp36 / tmp40
    tmp42 = -tmp41
    tmp44 = tl.where(tmp38, tmp43, tmp43)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp42 * tmp45
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp42 * tmp48
    tl.store(out_ptr0 + (x2), tmp24, xmask)
    tl.store(out_ptr1 + (x2), tmp46, xmask)
    tl.store(out_ptr2 + (x2), tmp49, xmask)
''')
