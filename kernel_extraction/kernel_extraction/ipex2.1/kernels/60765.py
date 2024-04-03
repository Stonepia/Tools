

# Original file: ./MobileBertForQuestionAnswering__0_backward_351.1/MobileBertForQuestionAnswering__0_backward_351.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ij/cijcjkdb4kpv3rp367l7l5uctb4otauzfs6lbuifr4bolujdnz2w.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.mul, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_3
triton_red_fused_embedding_dense_backward_mul_nll_loss_forward_sum_42 = async_compile.triton('triton_red_fused_embedding_dense_backward_mul_nll_loss_forward_sum_42', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_dense_backward_mul_nll_loss_forward_sum_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_embedding_dense_backward_mul_nll_loss_forward_sum_42(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x3 + (65536*r2)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    x1 = (xindex // 512)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.where(tmp6 < 0, tmp6 + 512, tmp6)
    tmp8 = tl.full([1, 1], -1, tl.int64)
    tmp9 = tmp6 == tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp4)
    tl.atomic_add(out_ptr1 + (x0 + (512*tmp7)), tmp11, None)
''')
