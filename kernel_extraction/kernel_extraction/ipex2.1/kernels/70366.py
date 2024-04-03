

# Original file: ./maml__21_backward_64.2/maml__21_backward_64.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/rn/crnzyn36l2frw252deqaarbranwvywzf5fjuhiq3e5rhfyzh7uy7.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.relu]
# batch_norm_3 => add_12, convert_element_type_15, rsqrt_3, var_mean_3
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_native_batch_norm_backward_relu_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (64 + x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (128 + x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (192 + x0), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (256 + x0), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (64 + x0), xmask).to(tl.float32)
    tmp21 = tl.load(in_ptr1 + (128 + x0), xmask).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (192 + x0), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr1 + (256 + x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp18 = triton_helpers.maximum(0, tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp16 + tmp19
    tmp22 = triton_helpers.maximum(0, tmp21)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 + tmp23
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp24 + tmp27
    tmp30 = triton_helpers.maximum(0, tmp29)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp28 + tmp31
    tmp33 = 5.0
    tmp34 = tmp32 / tmp33
    tmp35 = tmp16 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tmp19 - tmp34
    tmp38 = tmp37 * tmp37
    tmp39 = tmp36 + tmp38
    tmp40 = tmp23 - tmp34
    tmp41 = tmp40 * tmp40
    tmp42 = tmp39 + tmp41
    tmp43 = tmp27 - tmp34
    tmp44 = tmp43 * tmp43
    tmp45 = tmp42 + tmp44
    tmp46 = tmp31 - tmp34
    tmp47 = tmp46 * tmp46
    tmp48 = tmp45 + tmp47
    tmp49 = tmp48 / tmp33
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.rsqrt(tmp51)
    tmp53 = tmp1 * tmp35
    tmp54 = tmp3 * tmp37
    tmp55 = tmp53 + tmp54
    tmp56 = tmp6 * tmp40
    tmp57 = tmp55 + tmp56
    tmp58 = tmp9 * tmp43
    tmp59 = tmp57 + tmp58
    tmp60 = tmp12 * tmp46
    tmp61 = tmp59 + tmp60
    tmp62 = tmp61 * tmp52
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp34, xmask)
    tl.store(out_ptr2 + (x0), tmp52, xmask)
    tl.store(out_ptr3 + (x0), tmp61, xmask)
    tl.store(out_ptr4 + (x0), tmp62, xmask)
''')
