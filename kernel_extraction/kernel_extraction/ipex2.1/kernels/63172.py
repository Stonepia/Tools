

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/lp/clpwt4xdz7wuti7dxqqrotegladjw6upndis4n3jeqyis6j4dwtj.py
# Source Nodes: [add_118, mul_164, mul_167, pow_57], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.native_dropout_backward, aten.pow, aten.tanh_backward]
# add_118 => add_143
# mul_164 => mul_326
# mul_167 => mul_329
# pow_57 => convert_element_type_354
triton_poi_fused__to_copy_add_mul_native_dropout_backward_pow_tanh_backward_4 = async_compile.triton('triton_poi_fused__to_copy_add_mul_native_dropout_backward_pow_tanh_backward_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_native_dropout_backward_pow_tanh_backward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_native_dropout_backward_pow_tanh_backward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None)
    tmp17 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp6 * tmp18
    tmp20 = tmp19 * tmp10
    tmp21 = tmp11 * tmp11
    tmp22 = tmp12 - tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = 0.7978845608028654
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 0.044715
    tmp28 = tmp25 * tmp27
    tmp29 = tmp7.to(tl.float32)
    tmp30 = tmp29 * tmp29
    tmp31 = 3.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp28 * tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp26 + tmp34
    tmp36 = tmp19 * tmp13
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 * tmp8
    tmp39 = tmp35 + tmp38
    tl.store(out_ptr0 + (x0), tmp16, None)
    tl.store(out_ptr2 + (x0), tmp39, None)
''')
