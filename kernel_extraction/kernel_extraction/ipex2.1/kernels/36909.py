

# Original file: ./ElectraForCausalLM__0_backward_135.1/ElectraForCausalLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/z2/cz2gfqueo3gjyao4t7zh5xhm3x7bsh7sid2deohscjlehujch4mf.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_2
triton_per_fused_embedding_dense_backward_nll_loss_forward_sum_34 = async_compile.triton('triton_per_fused_embedding_dense_backward_nll_loss_forward_sum_34', '''
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
    size_hints=[65536, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_nll_loss_forward_sum_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_nll_loss_forward_sum_34(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 128)
    x2 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x0 + (65536*r1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.where(tmp5 < 0, tmp5 + 512, tmp5)
    tmp7 = tl.full([1, 1], -1, tl.int64)
    tmp8 = tmp5 == tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp9, tmp4)
    tl.atomic_add(out_ptr1 + (x2 + (128*tmp6)), tmp10, None)
''')
