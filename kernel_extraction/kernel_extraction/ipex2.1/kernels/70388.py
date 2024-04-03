

# Original file: ./maml__21_backward_64.2/maml__21_backward_64.2_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/rr/crrsttjl6tcd5hmuns6ozzqbnio7w2vi6feothveikurctzzcpnt.py
# Source Nodes: [batch_norm_2, cross_entropy, relu_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.convolution_backward, aten.native_batch_norm_backward, aten.nll_loss_forward, aten.relu, aten.threshold_backward]
# batch_norm_2 => convert_element_type_8
# cross_entropy => full_default_1
# relu_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_nll_loss_forward_relu_threshold_backward_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_nll_loss_forward_relu_threshold_backward_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_nll_loss_forward_relu_threshold_backward_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_nll_loss_forward_relu_threshold_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4) % 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp4 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp1.to(tl.float32)
    tmp8 = tmp6 - tmp7
    tmp10 = 0.05
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 - tmp15
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 - tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp12 * tmp21
    tmp23 = tmp19 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.where(tmp3, tmp2, tmp24)
    tl.store(in_out_ptr0 + (x3), tmp25, xmask)
''')
