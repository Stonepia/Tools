

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/oj/cojz4bjxnpif43kgdzfs3bu76dzu6kbtgj7uxehmhii2vvjo4ws4.py
# Source Nodes: [abs_5, add_15, add_16, iadd_2, max_3, mul_42, mul_43, mul_44, mul_45, neg_7, setitem_12, tanh_2, tensor_1, truediv_11], Original ATen: [aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.mul, aten.neg, aten.slice_scatter, aten.tanh]
# abs_5 => abs_5
# add_15 => add_17
# add_16 => add_18
# iadd_2 => add_19, slice_scatter_49, slice_scatter_50, slice_scatter_51, slice_scatter_52
# max_3 => maximum_2
# mul_42 => mul_42
# mul_43 => mul_43
# mul_44 => mul_44
# mul_45 => mul_45
# neg_7 => neg_7
# setitem_12 => copy_12, slice_scatter_54, slice_scatter_55, slice_scatter_56, slice_scatter_57
# tanh_2 => tanh_2
# tensor_1 => full_default_2
# truediv_11 => div_11
triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_17 = async_compile.triton('triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1066104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x0 = xindex % 26
    x4 = xindex
    x3 = xindex % 5304
    tmp57 = tl.load(in_ptr0 + (5304 + x4), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x2
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tmp5 & tmp11
    tmp13 = x0
    tmp14 = tmp13 >= tmp7
    tmp15 = tmp14 & tmp12
    tmp16 = tl.load(in_ptr0 + (5304 + x4), tmp15 & xmask, other=0.0)
    tmp17 = tl.where(tmp15, tmp16, 0.0)
    tmp18 = tl.load(in_ptr0 + (5304 + x4), tmp12 & xmask, other=0.0)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, 0.0)
    tmp21 = tl.load(in_ptr0 + (5304 + x4), tmp11 & xmask, other=0.0)
    tmp22 = tl.where(tmp5, tmp20, tmp21)
    tmp23 = tl.where(tmp11, tmp22, 0.0)
    tmp24 = tl.load(in_ptr0 + (5304 + x4), tmp5 & xmask, other=0.0)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr2 + (5304 + x4), tmp5 & xmask, other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr3 + (5304 + x4), tmp5 & xmask, other=0.0)
    tmp30 = tl.load(in_ptr4 + ((-52) + x3 + (5200*x2)), tmp5 & xmask, other=0.0)
    tmp31 = tl.abs(tmp30)
    tmp32 = -tmp31
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34 / tmp33
    tmp36 = libdevice.tanh(tmp35)
    tmp37 = 1.0
    tmp38 = tmp36 + tmp37
    tmp39 = 0.5
    tmp40 = tmp38 * tmp39
    tmp41 = tmp29 * tmp40
    tmp42 = 50.0
    tmp43 = triton_helpers.maximum(tmp42, tmp41)
    tmp44 = tmp28 * tmp43
    tmp45 = tmp25 + tmp44
    tmp46 = tl.where(tmp5, tmp45, 0.0)
    tmp47 = tmp5 & tmp10
    tmp48 = tmp14 & tmp47
    tmp49 = tl.load(in_ptr0 + (5304 + x4), tmp48 & xmask, other=0.0)
    tmp50 = tl.where(tmp48, tmp49, 0.0)
    tmp51 = tl.load(in_ptr0 + (5304 + x4), tmp47 & xmask, other=0.0)
    tmp52 = tl.where(tmp14, tmp50, tmp51)
    tmp53 = tl.where(tmp47, tmp52, 0.0)
    tmp54 = tl.load(in_ptr0 + (5304 + x4), tmp10 & xmask, other=0.0)
    tmp55 = tl.where(tmp5, tmp53, tmp54)
    tmp56 = tl.where(tmp10, tmp55, 0.0)
    tmp58 = tl.where(tmp10, tmp56, tmp57)
    tmp59 = tl.where(tmp5, tmp46, tmp58)
    tmp60 = tl.where(tmp11, tmp59, 0.0)
    tmp61 = tl.where(tmp10, tmp60, tmp25)
    tmp62 = tl.where(tmp5, tmp61, 0.0)
    tmp63 = tl.where(tmp10, tmp59, 0.0)
    tmp64 = tl.where(tmp10, tmp63, tmp58)
    tmp65 = tl.where(tmp5, tmp62, tmp64)
    tl.store(out_ptr0 + (x4), tmp59, xmask)
    tl.store(out_ptr1 + (x4), tmp65, xmask)
''')
