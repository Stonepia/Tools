

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ad/cadzhhalhi473usgxkdwdlfrls24hhfhjaot73hdxogztit7agrh.py
# Source Nodes: [add_7, add_8, getattr_l__mod___model___17___conv_block_6, getattr_l__mod___model___18___conv_block_6, l__mod___model_19, l__mod___model_20, l__mod___model_21], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.relu]
# add_7 => add_26
# add_8 => add_29
# getattr_l__mod___model___17___conv_block_6 => add_25, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__mod___model___18___conv_block_6 => add_28, mul_20, rsqrt_20, sub_20, var_mean_20
# l__mod___model_19 => convolution_21
# l__mod___model_20 => add_30, mul_21, rsqrt_21, sub_21, var_mean_21
# l__mod___model_21 => relu_12
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

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_convolution_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 16384.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp12 = triton_helpers.maximum(0, tmp11)
    tl.store(out_ptr0 + (x2), tmp12, None)
''')
