

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ql/cqlwqgp5ixtvcsitlen34webeq5zkmstmjwtbko3hn5irkx7njnj.py
# Source Nodes: [abs_10, add_39, add_40, iadd_7, max_8, min_8, mul_86, mul_95, mul_97, mul_98, neg_16, neg_17, sub_16, tanh_7, tensor, tensor_1, truediv_21, truediv_22], Original ATen: [aten.abs, aten.add, aten.div, aten.lift_fresh, aten.maximum, aten.minimum, aten.mul, aten.neg, aten.sub, aten.tanh]
# abs_10 => abs_10
# add_39 => add_46
# add_40 => add_47
# iadd_7 => add_48
# max_8 => maximum_7
# min_8 => minimum_7
# mul_86 => mul_86
# mul_95 => mul_95
# mul_97 => mul_97
# mul_98 => mul_98
# neg_16 => neg_16
# neg_17 => neg_17
# sub_16 => sub_16
# tanh_7 => tanh_7
# tensor => full_default_1
# tensor_1 => full_default_2
# truediv_21 => div_21
# truediv_22 => div_22
triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_30 = async_compile.triton('triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_30', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_abs_add_div_lift_fresh_maximum_minimum_mul_neg_sub_tanh_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5226)
    x1 = (xindex // 26) % 201
    x3 = xindex % 5226
    x0 = xindex % 26
    x4 = xindex
    tmp17 = tl.load(in_ptr0 + (10634 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr2 + (10634 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (10634 + x3 + (5304*x2)), xmask).to(tl.float32)
    tmp26 = tl.load(in_ptr4 + (x4), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr5 + (x4), xmask).to(tl.float32)
    tmp0 = 2 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 1 + x1
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp6 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp10 & tmp5
    tmp12 = tl.load(in_ptr0 + (10634 + x3 + (5304*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp11, tmp12, 0.0)
    tmp14 = tl.load(in_ptr0 + (10634 + x3 + (5304*x2)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp15, 0.0)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp27 = -tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = 0.0
    tmp32 = triton_helpers.minimum(tmp31, tmp30)
    tmp33 = 1e-20
    tmp34 = tmp32 - tmp33
    tmp35 = tmp28 / tmp34
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
    tmp46 = tmp25 * tmp45
    tmp47 = 50.0
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp23 * tmp48
    tmp50 = tmp19 + tmp49
    tl.store(out_ptr0 + (x4), tmp50, xmask)
''')
