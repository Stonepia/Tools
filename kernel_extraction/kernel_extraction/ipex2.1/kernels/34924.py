

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/j3/cj3tvrh25eg2q6h6n6tozjbpjqtmzwrinqsnyjklpgx67hidzbnw.py
# Source Nodes: [abs_18, add_65, add_66, iadd_15, min_13, mul_175, mul_176, mul_177, mul_178, mul_179, neg_32, neg_33, pow_8, sub_21, tanh_15, tensor, truediv_38, truediv_39], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.minimum, aten.mul, aten.neg, aten.pow, aten.sub, aten.tanh]
# abs_18 => abs_18
# add_65 => add_80
# add_66 => add_81
# iadd_15 => add_82
# min_13 => minimum_12
# mul_175 => mul_175
# mul_176 => mul_176
# mul_177 => mul_177
# mul_178 => mul_178
# mul_179 => mul_179
# neg_32 => neg_32
# neg_33 => neg_33
# pow_8 => pow_8
# sub_21 => sub_21
# tanh_15 => tanh_15
# tensor => full_default_1
# truediv_38 => div_38
# truediv_39 => div_39
triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_minimum_mul_neg_pow_sub_tanh_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5000)
    x0 = xindex % 25
    x1 = (xindex // 25) % 200
    x4 = (xindex // 25)
    x5 = xindex
    tmp24 = tl.load(in_ptr3 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr5 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp30 = tl.load(in_ptr6 + (x5), xmask).to(tl.float32)
    tmp33 = tl.load(in_ptr7 + (x5), xmask).to(tl.float32)
    tmp52 = tl.load(in_ptr8 + (10660 + x0 + (26*x1) + (5304*x2)), xmask).to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + (52 + x0 + (26*x1) + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = 2 + x1
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp10 < tmp3
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp5
    tmp15 = tl.load(in_ptr2 + (x0 + (26*x4)), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp5, tmp18, 0.0)
    tmp20 = tl.where(tmp5, tmp19, tmp17)
    tmp21 = tl.where(tmp5, tmp9, tmp20)
    tmp22 = tl.where(tmp5, tmp7, tmp21)
    tmp23 = tmp22.to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp31 = -tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = triton_helpers.minimum(tmp17, tmp34)
    tmp36 = 1e-20
    tmp37 = tmp35 - tmp36
    tmp38 = tmp32 / tmp37
    tmp39 = tl.abs(tmp38)
    tmp40 = -tmp39
    tmp41 = 0.001
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42 / tmp41
    tmp44 = libdevice.tanh(tmp43)
    tmp45 = 1.0
    tmp46 = tmp44 + tmp45
    tmp47 = 0.5
    tmp48 = tmp46 * tmp47
    tmp49 = tmp29 * tmp48
    tmp50 = tmp38 * tmp38
    tmp51 = tmp49 * tmp50
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp51 * tmp53
    tmp55 = tmp23 + tmp54
    tl.store(out_ptr0 + (x5), tmp55, xmask)
''')
