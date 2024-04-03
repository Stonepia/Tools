

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/5q/c5q3bkxlifblnoqxroxselokwhfu4lve675wmenavc2ylximmblk.py
# Source Nodes: [getattr_l__mod___model___10___conv_block_0, l__mod___model_8, l__mod___model_9], Original ATen: [aten._native_batch_norm_legit, aten.reflection_pad2d, aten.relu]
# getattr_l__mod___model___10___conv_block_0 => reflection_pad2d_1
# l__mod___model_8 => add_2, convert_element_type_4, convert_element_type_5, mul_2, rsqrt_2, sub_2, var_mean_2
# l__mod___model_9 => relu_2
triton_poi_fused__native_batch_norm_legit_reflection_pad2d_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_reflection_pad2d_relu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_reflection_pad2d_relu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_reflection_pad2d_relu_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16896)
    x1 = (xindex // 256) % 66
    x0 = xindex % 256
    x5 = xindex
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + x2
    tmp1 = tl.abs(tmp0)
    tmp2 = tl.full([1], 63, tl.int32)
    tmp3 = tmp2 - tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-1) + x1
    tmp7 = tl.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (256*tmp10) + (16384*tmp5)), xmask).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 - tmp13
    tmp16 = 4096.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = triton_helpers.maximum(0, tmp22)
    tl.store(out_ptr0 + (x5), tmp23, xmask)
''')
