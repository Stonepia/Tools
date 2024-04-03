

# Original file: ./MobileBertForMaskedLM__0_backward_354.1/MobileBertForMaskedLM__0_backward_354.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/no/cnos5n6qxnoffzm7j4it3hvkvwbhpwythq67ocv7r4ktbgkt4m7i.py
# Source Nodes: [add_10, add_11, add_12, add_3, add_6, add_7, add_8, add_9, mul_2, mul_4, mul_5, mul_6], Original ATen: [aten.add, aten.mul]
# add_10 => add_10
# add_11 => add_11
# add_12 => add_12
# add_3 => add_3
# add_6 => add_6
# add_7 => add_7
# add_8 => add_8
# add_9 => add_9
# mul_2 => mul_2
# mul_4 => mul_6
# mul_5 => mul_7
# mul_6 => mul_8
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr8 + (x2), None).to(tl.float32)
    tmp20 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp3 + tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp1 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp15 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tmp19 + tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp17 + tmp28
    tl.store(out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr1 + (x2), tmp29, None)
''')
