

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/vo/cvolvqarcrhcrodfvujgq37zvuvxfq2az7xkex2ig6mjmvmialic.py
# Source Nodes: [add_29, add_30, float_8, full_like_1, log_1, lt_1, min_2, min_3, mul_34, neg, sub_1, to_34, truediv_2, truediv_3, where_1, zeros_like], Original ATen: [aten._to_copy, aten.add, aten.div, aten.full_like, aten.log, aten.lt, aten.minimum, aten.mul, aten.neg, aten.sub, aten.where, aten.zeros_like]
# add_29 => add_36
# add_30 => add_37
# float_8 => convert_element_type_47
# full_like_1 => full_default_4
# log_1 => log_1
# lt_1 => lt_1
# min_2 => minimum_1
# min_3 => minimum_2
# mul_34 => mul_88
# neg => neg
# sub_1 => sub_1
# to_34 => convert_element_type_48
# truediv_2 => div_8
# truediv_3 => div_9
# where_1 => where_1
# zeros_like => full_default_3
triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_10 = async_compile.triton('triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_div_full_like_log_lt_minimum_mul_neg_sub_where_zeros_like_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = (-1)*(tl.minimum(0, x0 + ((-1)*x1), tl.PropagateNan.NONE))
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp0.to(tl.float32)
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tmp6 = tl.log(tmp5)
    tmp7 = 2.0794415416798357
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tmp10 + tmp1
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tl.where(tmp2, tmp0, tmp13)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tmp14 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''')
