

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/wp/cwpkoonlxpekwa5smiwg73j2gjio25fhsjlyhuhc5gtnj3revf3c.py
# Source Nodes: [add, getattr_l__mod___model___10___conv_block_6, l__mod___model_8, l__mod___model_9], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.relu]
# add => add_5
# getattr_l__mod___model___10___conv_block_6 => add_4, mul_4, rsqrt_4, sub_4, var_mean_4
# l__mod___model_8 => add_2, mul_2, rsqrt_2, sub_2, var_mean_2
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

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = triton_helpers.maximum(0, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp14 / tmp4
    tmp16 = tmp15 + tmp6
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp13 * tmp17
    tmp19 = tmp10 + tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')
