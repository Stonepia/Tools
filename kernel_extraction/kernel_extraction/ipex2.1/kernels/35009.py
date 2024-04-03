

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/6e/c6eyknzyg637kabjuesubsm5cskh2wywjt5tduwouwxipr4xsjx6.py
# Source Nodes: [abs_8, add_31, add_32, iadd_5, max_6, min_6, mul_66, mul_75, mul_77, mul_78, neg_12, neg_13, sub_14, tanh_5, tensor, tensor_1, truediv_17, truediv_18], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.sub, aten.tanh]
# abs_8 => abs_8
# add_31 => add_36
# add_32 => add_37
# iadd_5 => add_38
# max_6 => maximum_5
# min_6 => minimum_5
# mul_66 => mul_66
# mul_75 => mul_75
# mul_77 => mul_77
# mul_78 => mul_78
# neg_12 => neg_12
# neg_13 => neg_13
# sub_14 => sub_14
# tanh_5 => tanh_5
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_17 => div_17
# truediv_18 => div_18
triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_26 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1005000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5025)
    x1 = (xindex // 25) % 201
    x0 = xindex % 25
    x5 = xindex
    tmp23 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (10635 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (10635 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (x5), xmask).to(tl.float32)
    tmp35 = tl.load(in_ptr5 + (x5), xmask).to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = 1 + x0
    tmp13 = tmp12 >= tmp7
    tmp14 = tmp13 & tmp11
    tmp15 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp11, tmp18, 0.0)
    tmp20 = tl.load(in_ptr0 + (10635 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp21, 0.0)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tmp24.to(tl.float32)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp33 = -tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = 0.0
    tmp38 = triton_helpers.minimum(tmp37, tmp36)
    tmp39 = 1e-20
    tmp40 = tmp38 - tmp39
    tmp41 = tmp34 / tmp40
    tmp42 = tl.abs(tmp41)
    tmp43 = -tmp42
    tmp44 = 0.001
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 / tmp44
    tmp47 = libdevice.tanh(tmp46)
    tmp48 = 1.0
    tmp49 = tmp47 + tmp48
    tmp50 = 0.5
    tmp51 = tmp49 * tmp50
    tmp52 = tmp31 * tmp51
    tmp53 = 50.0
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tmp55 = tmp29 * tmp54
    tmp56 = tmp25 + tmp55
    tl.store(out_ptr0 + (x5), tmp56, xmask)
''')
