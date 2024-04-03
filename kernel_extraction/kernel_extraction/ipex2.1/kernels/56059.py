

# Original file: ./LayoutLMForSequenceClassification__0_backward_135.1/LayoutLMForSequenceClassification__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/cj/ccjmlggsc5inl7er3crvgjx26gyyk5cvomqmzw4ys7kxzo2elle3.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_5
triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp5 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp12 = tl.load(in_ptr5 + (x2), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tmp4 / tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp1 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.exp(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp11 - tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp0 + tmp18
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''')
