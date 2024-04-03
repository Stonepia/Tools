

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/qf/cqfzjbgwpaoqw5q5o3znxgpzdvqymacxoafmzvwgxgef7le2etsf.py
# Source Nodes: [abs_17, add_62, add_63, iadd_14, min_13, mul_165, mul_166, mul_167, mul_168, mul_169, neg_30, neg_31, pow_7, sub_21, tanh_14, tensor, truediv_36, truediv_37], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_17 => abs_17
# add_62 => add_76
# add_63 => add_77
# iadd_14 => add_78
# min_13 => minimum_12
# mul_165 => mul_165
# mul_166 => mul_166
# mul_167 => mul_167
# mul_168 => mul_168
# mul_169 => mul_169
# neg_30 => neg_30
# neg_31 => neg_31
# pow_7 => pow_7
# sub_21 => sub_21
# tanh_14 => tanh_14
# tensor => full_default_1
# truediv_36 => div_36
# truediv_37 => div_37
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x4 = (xindex // 25)
    x5 = xindex
    tmp21 = tl.load(in_ptr2 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr4 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp27 = tl.load(in_ptr5 + (x5), xmask).to(tl.float32)
    tmp30 = tl.load(in_ptr6 + (x5), xmask).to(tl.float32)
    tmp49 = tl.load(in_ptr7 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 2 + x1
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.load(in_ptr1 + (x0 + (26*x4)), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = 0.0
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp16, 0.0)
    tmp18 = tl.where(tmp5, tmp17, tmp15)
    tmp19 = tl.where(tmp5, tmp7, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp28 = -tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = triton_helpers.minimum(tmp15, tmp31)
    tmp33 = 1e-20
    tmp34 = tmp32 - tmp33
    tmp35 = tmp29 / tmp34
    tmp36 = tl.abs(tmp35)
    tmp37 = -tmp36
    tmp38 = 0.001
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39 / tmp38
    tmp41 = libdevice.tanh(tmp40)
    tmp42 = 1.0
    tmp43 = tmp41 + tmp42
    tmp44 = 0.5
    tmp45 = tmp43 * tmp44
    tmp46 = tmp26 * tmp45
    tmp47 = tmp35 * tmp35
    tmp48 = tmp46 * tmp47
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp48 * tmp50
    tmp52 = tmp20 + tmp51
    tl.store(out_ptr0 + (x5), tmp52, xmask)
''')
