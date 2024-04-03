

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/dc/cdc22aje6xwzeypcuyx7j77ihoerdwi5gepap4r77bbamthevynz.py
# Source Nodes: [abs_6, add_19, add_20, iadd_3, max_4, mul_43, mul_52, mul_54, mul_55, mul_58, neg_9, setitem_14, setitem_16, tanh_3, tensor_1, truediv_13, truediv_14], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.copy, aten.div, aten.lift_fresh, aten.maximum, aten.mul, aten.neg, aten.slice_scatter, aten.tanh]
# abs_6 => abs_6
# add_19 => add_22
# add_20 => add_23
# iadd_3 => add_24, convert_element_type_3, slice_scatter_62, slice_scatter_63, slice_scatter_64, slice_scatter_65
# max_4 => maximum_3
# mul_43 => mul_43
# mul_52 => mul_52
# mul_54 => mul_54
# mul_55 => mul_55
# mul_58 => mul_58
# neg_9 => neg_9
# setitem_14 => copy_14, slice_scatter_67, slice_scatter_68, slice_scatter_69, slice_scatter_70
# setitem_16 => copy_16, slice_scatter_75, slice_scatter_76
# tanh_3 => tanh_3
# tensor_1 => full_default_2
# truediv_13 => div_13
# truediv_14 => div_14
triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_18 = async_compile.triton('triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_abs_add_copy_div_lift_fresh_maximum_mul_neg_slice_scatter_tanh_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1066104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x4 = xindex
    x0 = xindex % 26
    x3 = xindex % 5304
    tmp51 = tl.load(in_ptr0 + (5304 + x4), xmask).to(tl.float32)
    tmp67 = tl.load(in_ptr6 + (5304 + x4), xmask).to(tl.float32)
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
    tmp13 = tl.load(in_ptr0 + (5304 + x4), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp12, tmp13, 0.0)
    tmp15 = tl.load(in_ptr0 + (5304 + x4), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.where(tmp11, tmp16, 0.0)
    tmp18 = tl.load(in_ptr0 + (5304 + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr2 + (5304 + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.load(in_ptr3 + (5304 + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.load(in_ptr4 + ((-52) + x3 + (5200*x2)), tmp5 & xmask, other=0.0)
    tmp28 = tl.abs(tmp27)
    tmp29 = -tmp28
    tmp30 = 0.001
    tmp31 = tmp29 + tmp30
    tmp32 = tmp31 / tmp30
    tmp33 = libdevice.tanh(tmp32)
    tmp34 = 1.0
    tmp35 = tmp33 + tmp34
    tmp36 = 0.5
    tmp37 = tmp35 * tmp36
    tmp38 = tmp26 * tmp37
    tmp39 = 50.0
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tmp41 = tmp24 * tmp40
    tmp42 = tmp20 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp5, tmp43, 0.0)
    tmp45 = tmp5 & tmp10
    tmp46 = tl.load(in_ptr0 + (5304 + x4), tmp45 & xmask, other=0.0).to(tl.float32)
    tmp47 = tl.where(tmp45, tmp46, 0.0)
    tmp48 = tl.load(in_ptr0 + (5304 + x4), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp49 = tl.where(tmp5, tmp47, tmp48)
    tmp50 = tl.where(tmp10, tmp49, 0.0)
    tmp52 = tl.where(tmp10, tmp50, tmp51)
    tmp53 = tl.where(tmp5, tmp44, tmp52)
    tmp54 = tl.where(tmp11, tmp53, 0.0)
    tmp55 = tl.where(tmp10, tmp54, tmp19)
    tmp56 = tl.where(tmp5, tmp55, 0.0)
    tmp57 = tl.where(tmp10, tmp53, 0.0)
    tmp58 = tl.where(tmp10, tmp57, tmp52)
    tmp59 = tl.where(tmp5, tmp56, tmp58)
    tmp60 = tl.where(tmp11, tmp59, 0.0)
    tmp61 = tl.where(tmp10, tmp60, tmp55)
    tmp62 = tl.load(in_ptr5 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp63 = 4.0
    tmp64 = tmp62 * tmp63
    tmp65 = tmp61 / tmp64
    tmp66 = tl.where(tmp5, tmp65, 0.0)
    tmp68 = tl.where(tmp5, tmp66, tmp67)
    tl.store(in_out_ptr0 + (x4), tmp68, xmask)
''')
