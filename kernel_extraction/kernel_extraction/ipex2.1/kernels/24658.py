

# Original file: ./MegatronBertForQuestionAnswering__0_backward_351.1/MegatronBertForQuestionAnswering__0_backward_351.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/up/cupojifub2ywalfuso5lsjtictrqk6ymqd7o73az4lwrhn7lhse3.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_3
triton_per_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_23 = async_compile.triton('triton_per_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_23', '''
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
    size_hints=[524288, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*i64', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_23(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 1024)
    x2 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x0 + (524288*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (524288*r1)), rmask)
    tmp10 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.where(tmp10 < 0, tmp10 + 512, tmp10)
    tmp12 = tl.full([1, 1], -1, tl.int64)
    tmp13 = tmp10 == tmp12
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp9)
    tl.atomic_add(out_ptr1 + (x2 + (1024*tmp11)), tmp15, None)
''')
