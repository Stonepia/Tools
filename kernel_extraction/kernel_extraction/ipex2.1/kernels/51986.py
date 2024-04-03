

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/c5/cc5a3umdtbnrl2gzgp2lxszlcwyuwio463erhf2dmmn2l7qtmqrx.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, conv2d, conv2d_1, conv2d_2, conv2d_3, relu, relu_1, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution, aten.relu]
# batch_norm => add, add_3, convert_element_type, convert_element_type_1, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_4, add_7, convert_element_type_4, convert_element_type_5, mul_25, mul_31, rsqrt_1, sub_19, var_mean_1
# batch_norm_2 => add_11, add_8, convert_element_type_8, convert_element_type_9, mul_32, mul_38, rsqrt_2, sub_20, var_mean_2
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# conv2d_3 => convolution_3
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_relu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_relu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 300.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')
