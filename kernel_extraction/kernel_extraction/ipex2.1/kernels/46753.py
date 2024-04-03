

# Original file: ./DistilBertForMaskedLM__0_backward_99.1/DistilBertForMaskedLM__0_backward_99.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/vr/cvrjqblpc3ghoppjrmiwu4r3q2tsg4akp4ncahmkb37hvgf4lt2r.py
# Source Nodes: [l__self___mlm_loss_fct], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward, aten.sum]
# l__self___mlm_loss_fct => full_default_7
triton_red_fused_embedding_dense_backward_nll_loss_forward_sum_31 = async_compile.triton('triton_red_fused_embedding_dense_backward_nll_loss_forward_sum_31', '''
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
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_dense_backward_nll_loss_forward_sum_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_embedding_dense_backward_nll_loss_forward_sum_31(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (98304*r1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    x3 = (xindex // 768)
    x2 = xindex % 768
    tmp4 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp5 = tl.where(tmp4 < 0, tmp4 + 512, tmp4)
    tmp6 = tl.full([1, 1], -1, tl.int64)
    tmp7 = tmp4 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp2)
    tl.atomic_add(out_ptr1 + (x2 + (768*tmp5)), tmp9, None)
''')
