

# Original file: ./maml__29_backward_89.15/maml__29_backward_89.15_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/2h/c2hbkm5qvhnghfxuw7ielmzrfkov6t3jdbhgdehv4qplxjjqtc55.py
# Source Nodes: [batch_norm], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# batch_norm => convert_element_type_3
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 169) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp0.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00591715976331361*(1/ks0)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tmp4 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.where(tmp2, tmp1, tmp23)
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''')
