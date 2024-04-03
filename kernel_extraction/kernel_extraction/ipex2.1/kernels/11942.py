

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/42/c42phgni2uhrrv72zjvrix235eqsaf7pb2kazjghcufmyeazm26a.py
# Source Nodes: [abs_4, abs_5, abs_6, add_13, gt_1, lt_2, maximum_4, maximum_5, minimum_2, minimum_3, mul_117, mul_118, mul_119, mul_120, mul_121, mul_122, mul_127, mul_128, mul_129, sub_35, sub_36, sub_37, sub_38, tensor, tensor_3, tensor_4, truediv_69, truediv_70, where_3, where_6, where_7], Original ATen: [aten.abs, aten.add, aten.div, aten.gt, aten.lift_fresh, aten.lt, aten.maximum, aten.minimum, aten.mul, aten.rsub, aten.scalar_tensor, aten.sub, aten.where]
# abs_4 => abs_4
# abs_5 => abs_5
# abs_6 => abs_6
# add_13 => add_65
# gt_1 => gt_1
# lt_2 => lt_2
# maximum_4 => maximum_4
# maximum_5 => maximum_5
# minimum_2 => minimum_2
# minimum_3 => minimum_3
# mul_117 => mul_120
# mul_118 => mul_121
# mul_119 => mul_122
# mul_120 => mul_123
# mul_121 => mul_124
# mul_122 => mul_125
# mul_127 => mul_130
# mul_128 => mul_131
# mul_129 => mul_132
# sub_35 => sub_35
# sub_36 => sub_36
# sub_37 => sub_37
# sub_38 => sub_38
# tensor => full_default
# tensor_3 => full_default_7
# tensor_4 => full_default_8
# truediv_69 => div_66
# truediv_70 => div_67
# where_3 => full_default_3
# where_6 => where_6
# where_7 => where_7
triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_70 = async_compile.triton('triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_70', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_70', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_70(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5226)
    x3 = xindex % 5226
    x1 = (xindex // 26) % 201
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (31902 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (31902 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (31824 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (32058 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), xmask).to(tl.float32)
    tmp41 = tl.load(in_ptr3 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tl.load(in_ptr4 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp57 = tl.load(in_ptr5 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 203, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tl.load(in_ptr2 + (10634 + x3 + (5304*x2)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (10608 + x3 + (5304*x2)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp8, tmp11, 0.0)
    tmp13 = tl.where(tmp8, tmp12, tmp1)
    tmp14 = tmp5 * tmp13
    tmp17 = tmp15 - tmp16
    tmp18 = 2 + x1
    tmp19 = tmp18 < tmp7
    tmp20 = tl.load(in_ptr2 + (10686 + x3 + (5304*x2)), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (10660 + x3 + (5304*x2)), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp19, tmp22, 0.0)
    tmp24 = tl.where(tmp19, tmp23, tmp1)
    tmp25 = tmp17 * tmp24
    tmp26 = tl.where(tmp2, tmp14, tmp25)
    tmp27 = tmp16 - tmp3
    tmp28 = 1 + x1
    tmp29 = tmp28 < tmp7
    tmp30 = tl.load(in_ptr2 + (10660 + x3 + (5304*x2)), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr2 + (10634 + x3 + (5304*x2)), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.where(tmp29, tmp32, 0.0)
    tmp34 = tl.where(tmp29, tmp33, tmp1)
    tmp35 = tmp27 * tmp34
    tmp36 = tl.abs(tmp35)
    tmp37 = 1e-20
    tmp38 = tmp36 < tmp37
    tmp39 = tl.where(tmp38, tmp1, tmp35)
    tmp40 = tmp26 / tmp39
    tmp42 = tmp41 * tmp0
    tmp43 = tl.abs(tmp42)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = 2.0
    tmp46 = tmp40 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = 1.0
    tmp49 = triton_helpers.minimum(tmp48, tmp47)
    tmp50 = tmp40.to(tl.float32)
    tmp51 = triton_helpers.minimum(tmp45, tmp50)
    tmp52 = triton_helpers.maximum(tmp49, tmp51)
    tmp53 = triton_helpers.maximum(tmp1, tmp52)
    tmp54 = tmp48 - tmp53
    tmp55 = tmp42 * tmp48
    tmp58 = tmp56 * tmp57
    tmp59 = tmp55 / tmp58
    tmp60 = tl.abs(tmp59)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61 * tmp53
    tmp63 = tmp54 + tmp62
    tmp64 = tmp44 * tmp63
    tmp65 = tmp35.to(tl.float32)
    tmp66 = tmp64 * tmp65
    tl.store(out_ptr0 + (x4), tmp66, xmask)
''')