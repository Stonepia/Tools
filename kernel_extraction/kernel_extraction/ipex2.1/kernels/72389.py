

# Original file: ./DebertaForQuestionAnswering__0_backward_135.1/DebertaForQuestionAnswering__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/xk/cxku3degddgqolvuqwrfbpo5fia6sssejrun3uexqn5rkxhxnbam.py
# Source Nodes: [trampoline_autograd_apply], Original ATen: [aten.add, aten.div, aten.embedding_dense_backward, aten.masked_fill, aten.sum]
# trampoline_autograd_apply => full_default_1
triton_per_fused_add_div_embedding_dense_backward_masked_fill_sum_32 = async_compile.triton('triton_per_fused_add_div_embedding_dense_backward_masked_fill_sum_32', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_dense_backward_masked_fill_sum_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_embedding_dense_backward_masked_fill_sum_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 768)
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x3 + (393216*r2)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (393216*r2)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (512*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (x1 + (512*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 768.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tl.where(tmp13 < 0, tmp13 + 512, tmp13)
    tmp15 = tl.full([1, 1], -1, tl.int64)
    tmp16 = tmp13 == tmp15
    tmp17 = 0.0
    tmp18 = tl.where(tmp16, tmp17, tmp12)
    tl.atomic_add(out_ptr1 + (x0 + (768*tmp14)), tmp18, None)
''')
