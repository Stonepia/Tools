

# Original file: ./MobileBertForQuestionAnswering__0_backward_351.1/MobileBertForQuestionAnswering__0_backward_351.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/e6/ce6hx775tijdznq2f5rayojgondubgxu32n4idrgzldo7heealds.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul]

triton_poi_fused__to_copy_add_mul_36 = async_compile.triton('triton_poi_fused__to_copy_add_mul_36', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_36', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')