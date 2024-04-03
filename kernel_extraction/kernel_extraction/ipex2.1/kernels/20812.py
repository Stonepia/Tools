

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/7u/c7uxl6g5nhednm6ypwpryachjqs4ebjebqsec422ak2riptqsa6k.py
# Source Nodes: [add_59, mul_45], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_59 => add_108
# mul_45 => mul_93
triton_poi_fused_add_mul_pow_tanh_backward_10 = async_compile.triton('triton_poi_fused_add_mul_pow_tanh_backward_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_backward_10(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.7978845608028654
    tmp14 = tmp12 * tmp13
    tmp15 = 0.044715
    tmp16 = tmp14 * tmp15
    tmp17 = tmp1 * tmp1
    tmp18 = 3.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp14 + tmp20
    tmp22 = tmp6 + tmp9
    tmp23 = tmp0 * tmp22
    tmp24 = tmp23 * tmp2
    tmp25 = tmp21 + tmp24
    tl.store(in_out_ptr0 + (x0), tmp25, None)
''')
