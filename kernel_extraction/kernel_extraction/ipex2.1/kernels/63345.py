

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/y5/cy5v6smiatyhazu23i373se2jjpavjiqo7b2esyqy6xlahjdoxtq.py
# Source Nodes: [add_118, mul_164, mul_167], Original ATen: [aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
# add_118 => add_143
# mul_164 => mul_326
# mul_167 => mul_329
triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_6 = async_compile.triton('triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i1', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp6 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tmp5 * tmp12
    tmp15 = tmp5 * tmp14
    tmp16 = tmp15 * tmp8
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp9.to(tl.float32)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 0.7978845608028654
    tmp24 = tmp22 * tmp23
    tmp25 = 0.044715
    tmp26 = tmp24 * tmp25
    tmp27 = tmp6 * tmp6
    tmp28 = 3.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tmp15 * tmp11
    tmp33 = tmp32 * tmp7
    tmp34 = tmp31 + tmp33
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(in_out_ptr0 + (x0), tmp34, None)
''')
