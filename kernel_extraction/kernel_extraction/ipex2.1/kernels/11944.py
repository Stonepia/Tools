

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/du/cduzlwp45kmk7sydrwwhj6m6ngfvboiymeo36ihvsbuil5vf5mb4.py
# Source Nodes: [mul_100, mul_145, mul_99, neg_54, setitem_104, sub_45, sub_46, sub_47, truediv_73, truediv_74], Original ATen: [aten.copy, aten.div, aten.mul, aten.neg, aten.sub]
# mul_100 => mul_103
# mul_145 => mul_148
# mul_99 => mul_102
# neg_54 => neg_54
# setitem_104 => copy_104
# sub_45 => sub_45
# sub_46 => sub_46
# sub_47 => sub_47
# truediv_73 => div_70
# truediv_74 => div_71
triton_poi_fused_copy_div_mul_neg_sub_72 = async_compile.triton('triton_poi_fused_copy_div_mul_neg_sub_72', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_mul_neg_sub_72', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_div_mul_neg_sub_72(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x3 = xindex % 5200
    x4 = xindex
    tmp50 = tl.load(in_ptr3 + (10660 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp52 = tl.load(in_ptr4 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tl.load(in_ptr5 + (2 + x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tl.load(in_ptr6 + (10660 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp57 = tl.load(in_ptr6 + (10634 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp59 = tl.load(in_ptr7 + (2 + x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2 + x1
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + (31980 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (47892 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (5200 + x4), tmp11 & xmask, other=0.0)
    tmp21 = tmp20 * tmp17
    tmp22 = tmp19 - tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.where(tmp11, tmp23, 0.0)
    tmp25 = 0.0
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tl.where(tmp5, tmp26, 0.0)
    tmp28 = tl.where(tmp5, tmp27, tmp25)
    tmp29 = 1 + x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp10 & tmp32
    tmp34 = tl.load(in_ptr0 + (16068 + (3*x3) + (15912*x2)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr1 + (31980 + (3*x3) + (15912*x2)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr1 + (16068 + (3*x3) + (15912*x2)), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 * tmp37
    tmp39 = tmp38 * tmp17
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.load(in_ptr2 + (x4), tmp33 & xmask, other=0.0)
    tmp42 = tmp41 * tmp17
    tmp43 = tmp40 - tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tl.where(tmp33, tmp44, 0.0)
    tmp46 = tl.where(tmp10, tmp45, tmp25)
    tmp47 = tl.where(tmp32, tmp46, 0.0)
    tmp48 = tl.where(tmp32, tmp47, tmp25)
    tmp49 = tmp28 - tmp48
    tmp51 = -tmp49
    tmp54 = tmp52 * tmp53
    tmp55 = tmp51 / tmp54
    tmp58 = tmp56 - tmp57
    tmp60 = tmp52 * tmp59
    tmp61 = tmp58 / tmp60
    tmp62 = tmp55 - tmp61
    tmp63 = tmp50 * tmp62
    tl.store(in_out_ptr0 + (x4), tmp63, xmask)
''')
