

# Original file: ./maml__22_forward_66.3/maml__22_forward_66.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/2a/c2asvzmh564filacei32bvftwgxkfwxgwehxdn5yrbkfeueaf7qx.py
# Source Nodes: [batch_norm, batch_norm_1, batch_norm_2, batch_norm_3, conv2d, conv2d_1, conv2d_2, conv2d_3, relu, relu_1, relu_2, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten._to_copy, aten.convolution, aten.relu]
# batch_norm => add, add_3, convert_element_type_3, convert_element_type_4, mul_18, mul_24, rsqrt, sub_18, var_mean
# batch_norm_1 => add_4, add_7, convert_element_type_7, convert_element_type_8, mul_25, mul_31, rsqrt_1, sub_19, var_mean_1
# batch_norm_2 => add_11, add_8, convert_element_type_11, convert_element_type_12, mul_32, mul_38, rsqrt_2, sub_20, var_mean_2
# batch_norm_3 => add_12, add_15, convert_element_type_15, convert_element_type_16, mul_39, mul_45, rsqrt_3, sub_21, var_mean_3
# conv2d => convert_element_type, convert_element_type_1, convert_element_type_2, convolution
# conv2d_1 => convert_element_type_5, convert_element_type_6, convolution_1
# conv2d_2 => convert_element_type_10, convert_element_type_9, convolution_2
# conv2d_3 => convert_element_type_13, convert_element_type_14, convolution_3
# relu => relu
# relu_1 => relu_1
# relu_2 => relu_2
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional__to_copy_convolution_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 - tmp5
    tmp8 = 75.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''')
