

# Original file: ./maml__29_backward_89.15/maml__29_backward_89.15.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/36/c36sea6gg3gt6yzfgq25eb72zkedbppvkf6jj24lyaqzgveb7xf4.py
# Source Nodes: [relu_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
# relu_1 => relu_1
triton_poi_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6 = async_compile.triton('triton_poi_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 36) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp6 = tmp1 - tmp5
    tmp8 = 0.0277777777777778*(1/ks0)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tl.where(tmp3, tmp2, tmp21)
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)
''')
