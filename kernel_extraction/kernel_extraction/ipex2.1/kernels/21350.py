

# Original file: ./maml__24_backward_74.6/maml__24_backward_74.6.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6s/c6s7tdtrydxd47gza7tnsy47xw3x2qkw3u2cni6bysxngnmpvzvm.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.relu]
# batch_norm_3 => add_12, rsqrt_3, var_mean_3
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp9 = tl.load(in_ptr1 + (x0), xmask)
    tmp11 = tl.load(in_ptr1 + (64 + x0), xmask)
    tmp14 = tl.load(in_ptr1 + (128 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (192 + x0), xmask)
    tmp20 = tl.load(in_ptr1 + (256 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = triton_helpers.maximum(0, tmp9)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tmp10 + tmp12
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tmp13 + tmp15
    tmp18 = triton_helpers.maximum(0, tmp17)
    tmp19 = tmp16 + tmp18
    tmp21 = triton_helpers.maximum(0, tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = 5.0
    tmp24 = tmp22 / tmp23
    tmp25 = tmp10 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tmp12 - tmp24
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp15 - tmp24
    tmp31 = tmp30 * tmp30
    tmp32 = tmp29 + tmp31
    tmp33 = tmp18 - tmp24
    tmp34 = tmp33 * tmp33
    tmp35 = tmp32 + tmp34
    tmp36 = tmp21 - tmp24
    tmp37 = tmp36 * tmp36
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38 / tmp23
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp0 * tmp25
    tmp44 = tmp1 * tmp27
    tmp45 = tmp43 + tmp44
    tmp46 = tmp3 * tmp30
    tmp47 = tmp45 + tmp46
    tmp48 = tmp5 * tmp33
    tmp49 = tmp47 + tmp48
    tmp50 = tmp7 * tmp36
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51 * tmp42
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
    tl.store(out_ptr2 + (x0), tmp42, xmask)
    tl.store(out_ptr3 + (x0), tmp51, xmask)
    tl.store(out_ptr4 + (x0), tmp52, xmask)
''')
