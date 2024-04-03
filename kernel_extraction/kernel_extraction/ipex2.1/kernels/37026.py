

# Original file: ./GPT2ForSequenceClassification__0_backward_135.1/GPT2ForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/2d/c2du3g6ncrxrvdcsczbi7v634pnckfqfrhkwnvam3f5t6vft2ajt.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.index_put, aten.new_zeros, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_25
triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2)
    x2 = xindex
    x0 = xindex % 2
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask)
    tmp5 = tl.load(in_ptr3 + (x2), xmask)
    tmp6 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp11 = tl.load(in_ptr6 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp17 = tl.load(in_ptr7 + (x2), xmask)
    tmp19 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 4, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 4)) | ~xmask, "index out of bounds: 0 <= tmp1 < 4")
    tmp3 = tl.where(tmp2 < 0, tmp2 + 1024, tmp2)
    # tl.device_assert(((0 <= tmp3) & (tmp3 < 1024)) | ~xmask, "index out of bounds: 0 <= tmp3 < 1024")
    tmp7 = tl.full([1], -100, tl.int64)
    tmp8 = tmp6 != tmp7
    tmp13 = tmp10 / tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tmp5 * tmp15
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp16 - tmp20
    tmp22 = tmp4 + tmp21
    tl.atomic_add(out_ptr0 + (x0 + (2*tmp3) + (2048*tmp1)), tmp22, xmask)
''')
