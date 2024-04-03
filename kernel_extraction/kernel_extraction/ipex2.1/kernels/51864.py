

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/yl/cylwlamyztvhyxdd6w2mgsxnd4hrea5aexlz2tlk7rnetbn2v74n.py
# Source Nodes: [add_7, add_8, getattr_l__self___model___17___conv_block_6, getattr_l__self___model___18___conv_block_6, l__self___model_19, l__self___model_20, l__self___model_21, l__self___model_22, l__self___model_23, l__self___model_24, l__self___model_25], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.reflection_pad2d, aten.relu]
# add_7 => add_26
# add_8 => add_29
# getattr_l__self___model___17___conv_block_6 => add_25, convert_element_type_75, convert_element_type_76, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__self___model___18___conv_block_6 => add_28, convert_element_type_83, convert_element_type_84, mul_20, rsqrt_20, sub_20, var_mean_20
# l__self___model_19 => convolution_21
# l__self___model_20 => add_30, convert_element_type_87, convert_element_type_88, mul_21, rsqrt_21, sub_21, var_mean_21
# l__self___model_21 => relu_12
# l__self___model_22 => convolution_22
# l__self___model_23 => add_31, convert_element_type_91, convert_element_type_92, mul_22, rsqrt_22, sub_22, var_mean_22
# l__self___model_24 => relu_13
# l__self___model_25 => reflection_pad2d_19
triton_poi_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16768)
    x1 = (xindex // 64) % 262
    x0 = xindex % 64
    x5 = xindex
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = (-3) + x2
    tmp1 = tl.abs(tmp0)
    tmp2 = tl.full([1], 255, tl.int32)
    tmp3 = tmp2 - tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-3) + x1
    tmp7 = tl.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (64*tmp10) + (16384*tmp5)), xmask).to(tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 - tmp15
    tmp18 = 65536.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp16 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = triton_helpers.maximum(0, tmp24)
    tl.store(out_ptr0 + (x5), tmp25, xmask)
''')
