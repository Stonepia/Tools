

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/sg/csglfjcwjupvp4valfwifq3u7lyqjvvnhys4kennvnt3pmecf2vo.py
# Source Nodes: [add_7, add_8, getattr_l__self___model___17___conv_block_6, getattr_l__self___model___18___conv_block_6, l__self___model_19, l__self___model_20, l__self___model_21, l__self___model_22, l__self___model_23], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.relu]
# add_7 => add_26
# add_8 => add_29
# getattr_l__self___model___17___conv_block_6 => add_25, convert_element_type_75, convert_element_type_76, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__self___model___18___conv_block_6 => add_28, convert_element_type_83, convert_element_type_84, mul_20, rsqrt_20, sub_20, var_mean_20
# l__self___model_19 => convolution_21
# l__self___model_20 => add_30, convert_element_type_87, convert_element_type_88, mul_21, rsqrt_21, sub_21, var_mean_21
# l__self___model_21 => relu_12
# l__self___model_22 => convolution_22
# l__self___model_23 => convert_element_type_91, var_mean_22
triton_per_fused__native_batch_norm_legit_add_convolution_relu_17 = async_compile.triton('triton_per_fused__native_batch_norm_legit_add_convolution_relu_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_add_convolution_relu_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_add_convolution_relu_17(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
    tl.store(out_ptr2 + (x3), tmp12, None)
''')
