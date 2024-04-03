

# Original file: ./DistillGPT2__0_backward_99.1/DistillGPT2__0_backward_99.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/rd/crd37o2wtkfxbg3vjwocdgipbmmk7l7xnchsgmyo52l5wyo5duaj.py
# Source Nodes: [add_23, mul_20, pow_6], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_23 => add_47
# mul_20 => mul_80
# pow_6 => convert_element_type_80
triton_poi_fused__to_copy_add_mul_pow_tanh_backward_11 = async_compile.triton('triton_poi_fused__to_copy_add_mul_pow_tanh_backward_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_pow_tanh_backward_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_mul_pow_tanh_backward_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp1 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp8
    tmp11 = tmp6 * tmp10
    tmp12 = 0.7978845608028654
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 0.044715
    tmp16 = tmp13 * tmp15
    tmp17 = tmp2.to(tl.float32)
    tmp18 = tmp17 * tmp17
    tmp19 = 3.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp14 + tmp22
    tmp24 = tmp7 + tmp9
    tmp25 = tmp1 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp3
    tmp28 = tmp23 + tmp27
    tl.store(in_out_ptr0 + (x0), tmp28, None)
''')
