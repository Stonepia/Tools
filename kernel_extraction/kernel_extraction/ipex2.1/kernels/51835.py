

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/eq/ceqrf3vsn7vlkvxabk22soajwwqxvdpcifbttq52a5bllwlhfv5g.py
# Source Nodes: [add, getattr_l__mod___model___10___conv_block_6, l__mod___model_8, l__mod___model_9], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.relu]
# add => add_5
# getattr_l__mod___model___10___conv_block_6 => add_4, convert_element_type_8, convert_element_type_9, mul_4, rsqrt_4, sub_4, var_mean_4
# l__mod___model_8 => add_2, convert_element_type_4, convert_element_type_5, mul_2, rsqrt_2, sub_2, var_mean_2
# l__mod___model_9 => relu_2
triton_poi_fused__native_batch_norm_legit_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_relu_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp5
    tmp19 = tmp18 + tmp7
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp12 + tmp22
    tl.store(out_ptr0 + (x2), tmp23, None)
''')
