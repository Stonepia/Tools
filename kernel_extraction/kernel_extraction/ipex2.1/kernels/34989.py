

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/db/cdbei3l4smgbcwq24tlnqtamcqvyty2qfww622byndvhck5fbnss.py
# Source Nodes: [abs_13, add_49, add_50, iadd_10, iadd_9, min_9, mul_125, mul_126, mul_127, mul_128, mul_129, neg_22, neg_23, pow_3, sub_17, tanh_10, tensor, truediv_28, truediv_29], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.slice_scatter, aten.sub, aten.tanh]
# abs_13 => abs_13
# add_49 => add_59
# add_50 => add_60
# iadd_10 => add_61, convert_element_type_10, slice_scatter_150
# iadd_9 => convert_element_type_9, slice_scatter_144
# min_9 => minimum_8
# mul_125 => mul_125
# mul_126 => mul_126
# mul_127 => mul_127
# mul_128 => mul_128
# mul_129 => mul_129
# neg_22 => neg_22
# neg_23 => neg_23
# pow_3 => pow_3
# sub_17 => sub_17
# tanh_10 => tanh_10
# tensor => full_default_1
# truediv_28 => div_28
# truediv_29 => div_29
triton_poi_fused__to_copy_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_6 = async_compile.triton('triton_poi_fused__to_copy_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_abs_add_div_lift_fresh_minimum_mul_neg_pow_slice_scatter_sub_tanh_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x4 = xindex % 5200
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (25*x3)), tmp2 & xmask, other=0.0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.where(tmp2, tmp4, 0.0)
    tmp6 = 2 + x2
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1], 202, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = 2 + x1
    tmp13 = tmp12 >= tmp7
    tmp14 = tmp12 < tmp9
    tmp15 = tmp13 & tmp14
    tmp16 = tmp15 & tmp11
    tmp17 = tmp2 & tmp16
    tmp18 = tl.load(in_ptr1 + (1 + x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (10660 + x4 + (5304*x2)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0 + (25*x3)), tmp17 & xmask, other=0.0)
    tmp23 = tl.abs(tmp22)
    tmp24 = -tmp23
    tmp25 = 0.001
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 / tmp25
    tmp28 = libdevice.tanh(tmp27)
    tmp29 = 1.0
    tmp30 = tmp28 + tmp29
    tmp31 = 0.5
    tmp32 = tmp30 * tmp31
    tmp33 = tmp21 * tmp32
    tmp34 = tmp22 * tmp22
    tmp35 = tmp33 * tmp34
    tmp36 = tl.load(in_ptr4 + (10660 + x4 + (5304*x2)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 * tmp37
    tmp39 = 0.0
    tmp40 = tmp39 + tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.where(tmp17, tmp41, 0.0)
    tmp43 = tl.where(tmp2, tmp42, tmp39)
    tmp44 = tl.where(tmp16, tmp43, 0.0)
    tmp45 = tl.where(tmp15, tmp44, tmp39)
    tmp46 = tl.where(tmp11, tmp45, 0.0)
    tmp47 = tl.where(tmp11, tmp46, tmp39)
    tmp48 = tl.where(tmp2, tmp5, tmp47)
    tmp49 = tl.load(in_ptr5 + (1 + x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.load(in_ptr6 + (1 + x1), tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tmp49 * tmp50
    tmp52 = tl.load(in_ptr2 + (10660 + x4 + (5304*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr7 + (x0 + (25*x3)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp56 = -tmp55
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tl.load(in_ptr8 + (x0 + (25*x3)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = triton_helpers.minimum(tmp39, tmp59)
    tmp61 = 1e-20
    tmp62 = tmp60 - tmp61
    tmp63 = tmp57 / tmp62
    tmp64 = tl.abs(tmp63)
    tmp65 = -tmp64
    tmp66 = tmp65 + tmp25
    tmp67 = tmp66 / tmp25
    tmp68 = libdevice.tanh(tmp67)
    tmp69 = tmp68 + tmp29
    tmp70 = tmp69 * tmp31
    tmp71 = tmp54 * tmp70
    tmp72 = tmp63 * tmp63
    tmp73 = tmp71 * tmp72
    tmp74 = tl.load(in_ptr4 + (10660 + x4 + (5304*x2)), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp73 * tmp75
    tmp77 = tmp39 + tmp76
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tl.where(tmp2, tmp78, 0.0)
    tmp80 = tl.where(tmp2, tmp79, tmp39)
    tl.store(out_ptr0 + (x5), tmp48, xmask)
    tl.store(out_ptr1 + (x5), tmp80, xmask)
''')
