

# Original file: ./GPT2ForSequenceClassification__0_backward_135.1/GPT2ForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/tg/ctgddllgaa6cwllc7ecfxygfubw2knd4k3dlqtytldhbhokdcxvx.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.index_put, aten.new_zeros, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_25
triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_add_index_put_new_zeros_nll_loss_backward_nll_loss_forward_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')