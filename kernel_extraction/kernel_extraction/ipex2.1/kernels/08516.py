

# Original file: ./DistillGPT2__0_backward_99.1/DistillGPT2__0_backward_99.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/rv/crv3k5wusgqr4kkpdfdeypfb4iosa6sqquoxh32owtcvtuxbpkbw.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.sum]

triton_per_fused_embedding_dense_backward_sum_23 = async_compile.triton('triton_per_fused_embedding_dense_backward_sum_23', '''
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
    size_hints=[524288, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_sum_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_sum_23(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 768)
    x2 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x0 + (393216*r1)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.where(tmp5 < 0, tmp5 + 1024, tmp5)
    tmp7 = tmp4.to(tl.float32)
    tmp8 = tl.full([1, 1], False, tl.int1)
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp9, tmp7)
    tl.atomic_add(out_ptr1 + (x2 + (768*tmp6)), tmp10, None)
''')
