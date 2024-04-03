

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/jt/cjt6oyxwyt6fehekymdu3y74w5m57mzm2txh6pu5ntpluhjlqhqp.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.relu]
# batch_norm_3 => add_13, add_14, convert_element_type_12, convert_element_type_14, convert_element_type_15, mul_22, mul_23, mul_24, mul_25, mul_26, var_mean_3
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional__to_copy_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional__to_copy_relu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__to_copy_relu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_relu_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (64 + x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (128 + x0), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (192 + x0), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (256 + x0), xmask).to(tl.float32)
    tmp38 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp47 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = triton_helpers.maximum(0, tmp3)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp2 + tmp5
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp6 + tmp9
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 + tmp13
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp14 + tmp17
    tmp19 = 5.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp2 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp5 - tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp9 - tmp20
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp13 - tmp20
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 + tmp30
    tmp32 = tmp17 - tmp20
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34 / tmp19
    tmp36 = 0.1
    tmp37 = tmp20 * tmp36
    tmp39 = 0.9
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp37 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 1.25
    tmp45 = tmp35 * tmp44
    tmp46 = tmp45 * tmp36
    tmp48 = tmp47 * tmp39
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp46 + tmp49
    tmp51 = tmp50.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp35, xmask)
    tl.store(out_ptr2 + (x0), tmp43, xmask)
    tl.store(out_ptr3 + (x0), tmp51, xmask)
''')
