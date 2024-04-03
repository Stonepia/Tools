

# Original file: ./LayoutLMForSequenceClassification__0_backward_135.1/LayoutLMForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/bj/cbjpafjer6kvadmwxtpyt6h3aqsvmej3n4xx2vmvuhx5xdtyzsz6.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_5
triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i1', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1 + (2*x0)), xmask).to(tl.float32)
    tmp6 = tmp3 / tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp1, tmp6, tmp7)
    tmp9 = tmp0 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp11 * tmp8
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''')
