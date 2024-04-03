

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/mp/cmphxjnse63bhhawijzxqh45g4l6yvspm6hszbfhwqswzywxuwp5.py
# Source Nodes: [abs_1, add_1, add_2, float_1, full_like, gt, iadd, log, lt, min_1, mul_3, mul_4, sub_1, to_3, to_4, truediv, truediv_1, where], Original ATen: [aten._to_copy, aten.abs, aten.add, aten.div, aten.full_like, aten.gt, aten.log, aten.lt, aten.minimum, aten.mul, aten.sub, aten.where]
# abs_1 => abs_1
# add_1 => add_1
# add_2 => add_2
# float_1 => convert_element_type_4
# full_like => full_default_1
# gt => gt_1
# iadd => add_3
# log => log
# lt => lt
# min_1 => minimum
# mul_3 => mul_5
# mul_4 => mul_6
# sub_1 => sub_1
# to_3 => convert_element_type_3
# to_4 => convert_element_type_5
# truediv => div
# truediv_1 => div_1
# where => where
triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3 = async_compile.triton('triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_abs_add_div_full_like_gt_log_lt_minimum_mul_sub_where_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1], 16, tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp1
    tmp7 = tl.abs(tmp0)
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp7.to(tl.float32)
    tmp11 = 8.0
    tmp12 = tmp10 / tmp11
    tmp13 = tl.log(tmp12)
    tmp14 = 2.772588722239781
    tmp15 = tmp13 / tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tmp17 + tmp8
    tmp19 = tl.full([1], 15, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp7, tmp20)
    tmp22 = tmp6 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
''')
