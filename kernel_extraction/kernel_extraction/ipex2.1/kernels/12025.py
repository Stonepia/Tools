

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/jp/cjpxs5k724g2lrid67q6u3bbkfrfzkze3jbtiew4zfgaodcrjoy4.py
# Source Nodes: [abs_1, abs_2, abs_3, add_11, gt, lt_1, maximum_2, maximum_3, minimum, minimum_1, mul_103, mul_104, mul_105, mul_106, mul_107, mul_108, mul_111, mul_112, mul_113, sub_30, sub_31, sub_32, sub_33, tensor, tensor_3, tensor_4, truediv_67, truediv_68, where_4, where_5], Original ATen: [aten.abs, aten.add, aten.div, aten.gt, aten.lift_fresh, aten.lt, aten.maximum, aten.minimum, aten.mul, aten.rsub, aten.scalar_tensor, aten.sub, aten.where]
# abs_1 => abs_1
# abs_2 => abs_2
# abs_3 => abs_3
# add_11 => add_63
# gt => gt
# lt_1 => lt_1
# maximum_2 => maximum_2
# maximum_3 => maximum_3
# minimum => minimum
# minimum_1 => minimum_1
# mul_103 => mul_106
# mul_104 => mul_107
# mul_105 => mul_108
# mul_106 => mul_109
# mul_107 => mul_110
# mul_108 => mul_111
# mul_111 => mul_114
# mul_112 => mul_115
# mul_113 => mul_116
# sub_30 => sub_30
# sub_31 => sub_31
# sub_32 => sub_32
# sub_33 => sub_33
# tensor => full_default
# tensor_3 => full_default_7
# tensor_4 => full_default_8
# truediv_67 => div_64
# truediv_68 => div_65
# where_4 => where_4
# where_5 => full_default_5, where_5
triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_69 = async_compile.triton('triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_69', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_69', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_gt_lift_fresh_lt_maximum_minimum_mul_rsub_scalar_tensor_sub_where_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 5200
    x1 = (xindex // 5200)
    x2 = xindex
    x4 = (xindex // 26) % 200
    tmp0 = tl.load(in_ptr0 + (16068 + (3*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (16068 + (3*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (156 + (3*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (47892 + (3*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (31980 + (3*x0) + (15912*x1)), xmask).to(tl.float32)
    tmp55 = tl.load(in_ptr3 + (2 + x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tl.load(in_ptr4 + (1 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 203, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tl.load(in_ptr2 + (5356 + x0 + (5304*x1)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (52 + x0 + (5304*x1)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.where(tmp8, tmp11, 0.0)
    tmp13 = tl.where(tmp8, tmp12, tmp1)
    tmp14 = tmp5 * tmp13
    tmp17 = tmp15 - tmp16
    tmp18 = 2 + x1
    tmp19 = tmp18 < tmp7
    tmp20 = tl.load(in_ptr2 + (15964 + x0 + (5304*x1)), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (10660 + x0 + (5304*x1)), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp19, tmp22, 0.0)
    tmp24 = tl.where(tmp19, tmp23, tmp1)
    tmp25 = tmp17 * tmp24
    tmp26 = tl.where(tmp2, tmp14, tmp25)
    tmp27 = tmp16 - tmp3
    tmp28 = 1 + x1
    tmp29 = tmp28 < tmp7
    tmp30 = tl.load(in_ptr2 + (10660 + x0 + (5304*x1)), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr2 + (5356 + x0 + (5304*x1)), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.where(tmp29, tmp32, 0.0)
    tmp34 = tl.where(tmp29, tmp33, tmp1)
    tmp35 = tmp27 * tmp34
    tmp36 = tl.abs(tmp35)
    tmp37 = 1e-20
    tmp38 = tmp36 < tmp37
    tmp39 = 1.0005576689441423e-20
    tmp40 = tl.where(tmp38, tmp39, tmp35)
    tmp41 = tmp26 / tmp40
    tmp42 = tl.abs(tmp0)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 2.0
    tmp45 = tmp41 * tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = 1.0
    tmp48 = triton_helpers.minimum(tmp47, tmp46)
    tmp49 = tmp41.to(tl.float32)
    tmp50 = triton_helpers.minimum(tmp44, tmp49)
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tmp52 = triton_helpers.maximum(tmp1, tmp51)
    tmp53 = tmp47 - tmp52
    tmp54 = tmp0 * tmp47
    tmp57 = tmp55 * tmp56
    tmp58 = tmp54 / tmp57
    tmp59 = tl.abs(tmp58)
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp52
    tmp62 = tmp53 + tmp61
    tmp63 = tmp43 * tmp62
    tmp64 = tmp35.to(tl.float32)
    tmp65 = tmp63 * tmp64
    tl.store(out_ptr0 + (x2), tmp65, xmask)
''')
