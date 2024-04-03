

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/b2/cb2x4c7y2nmwbgwxi3jyuvdmkoafyjredezfln6remhtwlc2zupb.py
# Source Nodes: [add_7, add_8, getattr_l__self___model___17___conv_block_6, getattr_l__self___model___18___conv_block_6, l__self___model_19, l__self___model_20, l__self___model_21], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.relu]
# add_7 => add_26
# add_8 => add_29
# getattr_l__self___model___17___conv_block_6 => add_25, convert_element_type_75, convert_element_type_76, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__self___model___18___conv_block_6 => add_28, convert_element_type_83, convert_element_type_84, mul_20, rsqrt_20, sub_20, var_mean_20
# l__self___model_19 => convolution_21
# l__self___model_20 => add_30, convert_element_type_87, convert_element_type_88, mul_21, rsqrt_21, sub_21, var_mean_21
# l__self___model_21 => relu_12
triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 16384.0
    tmp8 = tmp6 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')
