

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/el/celpkqoodxsqur7obvcz4pbuzpgo3f3ojgyvcxxc6pvqzjpg2ox3.py
# Source Nodes: [add_1, getattr_l__self___model___11___conv_block_6, getattr_l__self___model___12___conv_block_0], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.reflection_pad2d]
# add_1 => add_8
# getattr_l__self___model___11___conv_block_6 => add_7, convert_element_type_27, convert_element_type_28, mul_6, rsqrt_6, sub_6, var_mean_6
# getattr_l__self___model___12___conv_block_0 => reflection_pad2d_5
triton_poi_fused__native_batch_norm_legit_add_reflection_pad2d_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_reflection_pad2d_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_reflection_pad2d_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_reflection_pad2d_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16896)
    x1 = (xindex // 256) % 66
    x0 = xindex % 256
    x5 = xindex
    tmp14 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp12 = tl.load(in_ptr1 + (x0 + (256*tmp10) + (16384*tmp5)), xmask).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp13 - tmp14
    tmp17 = 4096.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp11 + tmp23
    tl.store(out_ptr0 + (x5), tmp24, xmask)
''')
