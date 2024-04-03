

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ll/cllwuy4hgd5dov74sbgz5yvbli745jyw7ln7tnl657rcs2wpoqh2.py
# Source Nodes: [abs_6, add_19, add_20, iadd_3, max_4, mul_43, mul_52, mul_54, mul_55, neg_9, tanh_3, tensor_1, truediv_13], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.mul, aten.neg, aten.tanh]
# abs_6 => abs_6
# add_19 => add_22
# add_20 => add_23
# iadd_3 => add_24
# max_4 => maximum_3
# mul_43 => mul_43
# mul_52 => mul_52
# mul_54 => mul_54
# mul_55 => mul_55
# neg_9 => neg_9
# tanh_3 => tanh_3
# tensor_1 => full_default_2
# truediv_13 => div_13
triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_tanh_18 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_tanh_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_tanh_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_maximum_mul_neg_tanh_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5200)
    x3 = xindex % 5200
    x1 = (xindex // 26) % 200
    x0 = xindex % 26
    x4 = xindex
    tmp27 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), xmask)
    tmp31 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (5356 + x3 + (5304*x2)), xmask)
    tmp34 = tl.load(in_ptr5 + (5356 + x3 + (5304*x2)), xmask)
    tmp35 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (52 + x3 + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = tl.load(in_ptr1 + (52 + x3 + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = 2 + x1
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp10 < tmp3
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp5
    tmp16 = x0
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp18 & xmask, other=0.0)
    tmp20 = tl.where(tmp18, tmp19, 0.0)
    tmp21 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp15 & xmask, other=0.0)
    tmp22 = tl.where(tmp17, tmp20, tmp21)
    tmp23 = tl.where(tmp15, tmp22, 0.0)
    tmp24 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp5 & xmask, other=0.0)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp5, tmp25, 0.0)
    tmp28 = tl.where(tmp5, tmp26, tmp27)
    tmp29 = tl.where(tmp5, tmp9, tmp28)
    tmp30 = tl.where(tmp5, tmp7, tmp29)
    tmp33 = tmp31 * tmp32
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
    tmp46 = tmp34 * tmp45
    tmp47 = 50.0
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp33 * tmp48
    tmp50 = tmp30 + tmp49
    tl.store(in_out_ptr0 + (x4), tmp50, xmask)
''')
