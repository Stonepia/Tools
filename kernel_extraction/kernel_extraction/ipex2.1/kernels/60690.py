

# Original file: ./MobileBertForQuestionAnswering__0_backward_351.1/MobileBertForQuestionAnswering__0_backward_351.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/sx/csx2xf6quap26653eryi4b3t72frnvbiw3yxlqtthwbusumoefx4.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.threshold_backward]
# cross_entropy => full_default_3
triton_poi_fused_nll_loss_forward_threshold_backward_7 = async_compile.triton('triton_poi_fused_nll_loss_forward_threshold_backward_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_threshold_backward_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_nll_loss_forward_threshold_backward_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(in_out_ptr0 + (x0), tmp3, None)
''')
