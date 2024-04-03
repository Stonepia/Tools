

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/52/c52u2tvkgdpexnyejrojzh2bsv63gae3qlcyukzbg7hmawzvdabe.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# batch_norm_3 => add_13, add_14, mul_22, mul_23, mul_24, mul_25, mul_26, var_mean_3
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_relu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_relu_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp8 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp33 = tl.load(in_ptr1 + (x0), xmask)
    tmp40 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tmp1 + tmp3
    tmp6 = triton_helpers.maximum(0, tmp5)
    tmp7 = tmp4 + tmp6
    tmp9 = triton_helpers.maximum(0, tmp8)
    tmp10 = tmp7 + tmp9
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tmp10 + tmp12
    tmp14 = 5.0
    tmp15 = tmp13 / tmp14
    tmp16 = tmp1 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp3 - tmp15
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = tmp6 - tmp15
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tmp9 - tmp15
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tmp12 - tmp15
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29 / tmp14
    tmp31 = 0.1
    tmp32 = tmp15 * tmp31
    tmp34 = 0.9
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 + tmp35
    tmp37 = 1.25
    tmp38 = tmp30 * tmp37
    tmp39 = tmp38 * tmp31
    tmp41 = tmp40 * tmp34
    tmp42 = tmp39 + tmp41
    tl.store(out_ptr0 + (x0), tmp15, xmask)
    tl.store(out_ptr1 + (x0), tmp30, xmask)
    tl.store(out_ptr2 + (x0), tmp36, xmask)
    tl.store(out_ptr3 + (x0), tmp42, xmask)
''')
