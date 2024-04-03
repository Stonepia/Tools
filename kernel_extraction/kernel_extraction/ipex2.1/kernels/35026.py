

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/hz/chzk5ndwambclg76fsdvvts3uvjgkaqylz2gkb4hdjbsalh4grub.py
# Source Nodes: [abs_14, add_52, add_53, iadd_11, min_9, mul_135, mul_136, mul_137, mul_138, mul_139, neg_24, neg_25, pow_4, sub_17, tanh_11, tensor, truediv_30, truediv_31], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_14 => abs_14
# add_52 => add_63
# add_53 => add_64
# iadd_11 => add_65
# min_9 => minimum_8
# mul_135 => mul_135
# mul_136 => mul_136
# mul_137 => mul_137
# mul_138 => mul_138
# mul_139 => mul_139
# neg_24 => neg_24
# neg_25 => neg_25
# pow_4 => pow_4
# sub_17 => sub_17
# tanh_11 => tanh_11
# tensor => full_default_1
# truediv_30 => div_30
# truediv_31 => div_31
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_43 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_43', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_43', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x1 = (xindex // 25) % 200
    x0 = xindex % 25
    x4 = (xindex // 25)
    x5 = xindex
    tmp18 = tl.load(in_ptr1 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr2 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr3 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr4 + (x5), xmask).to(tl.float32)
    tmp27 = tl.load(in_ptr5 + (x5), xmask).to(tl.float32)
    tmp46 = tl.load(in_ptr6 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = tl.load(in_ptr0 + (x0 + (26*x4)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = 0.0
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp14, 0.0)
    tmp16 = tl.where(tmp5, tmp15, tmp13)
    tmp17 = tmp16.to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp25 = -tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = triton_helpers.minimum(tmp13, tmp28)
    tmp30 = 1e-20
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 / tmp31
    tmp33 = tl.abs(tmp32)
    tmp34 = -tmp33
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 / tmp35
    tmp38 = libdevice.tanh(tmp37)
    tmp39 = 1.0
    tmp40 = tmp38 + tmp39
    tmp41 = 0.5
    tmp42 = tmp40 * tmp41
    tmp43 = tmp23 * tmp42
    tmp44 = tmp32 * tmp32
    tmp45 = tmp43 * tmp44
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp45 * tmp47
    tmp49 = tmp17 + tmp48
    tl.store(out_ptr0 + (x5), tmp49, xmask)
''')
