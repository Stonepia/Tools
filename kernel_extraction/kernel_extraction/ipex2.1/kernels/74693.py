

# Original file: ./maml__29_backward_89.15/maml__29_backward_89.15_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/pd/cpd76gmgjnzsffga7mepsn73vn4vttuf7mrbb7h7fhbj2xxclm4m.py
# Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
# batch_norm_3 => convert_element_type_15
# relu_3 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp4 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp1.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 1.00000000000000*(1/ks0)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = tmp5 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.where(tmp3, tmp2, tmp24)
    tl.store(in_out_ptr0 + (x2), tmp25, xmask)
''')
