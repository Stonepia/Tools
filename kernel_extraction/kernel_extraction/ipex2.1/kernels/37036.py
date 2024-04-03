

# Original file: ./GPT2ForSequenceClassification__0_backward_135.1/GPT2ForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/zt/cztow4whx3tms5lpzgeuh4fm6whmzl35b3wlo55qrwtcchhhjwmt.py
# Source Nodes: [cross_entropy, full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.nll_loss_forward, aten.where]
# cross_entropy => full_default_25
# full => full_default
triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_14 = async_compile.triton('triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_nll_loss_forward_where_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 49152
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x2)), rmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp6 * tmp11
    tmp14 = tmp7 - tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp18, rmask)
''')
