

# Original file: ./MT5ForConditionalGeneration__0_backward_207.1/MT5ForConditionalGeneration__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/de/cdecuzxsbnsgzt6f5hkxtglhsbxwk4kh7wvtwetaz336itywqwwe.py
# Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.embedding_dense_backward, aten.sum]

triton_per_fused_add_clone_embedding_dense_backward_sum_26 = async_compile.triton('triton_per_fused_add_clone_embedding_dense_backward_sum_26', '''
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
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_embedding_dense_backward_sum_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_embedding_dense_backward_sum_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
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
    x1 = (xindex // 128)
    x4 = xindex % 16384
    x5 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x3 + (98304*r2)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + (98304*r2)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x3 + (98304*r2)), rmask)
    tmp8 = tl.load(in_ptr3 + (x3 + (98304*r2)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x1 + (768*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp0 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tl.where(tmp19 < 0, tmp19 + 32, tmp19)
    tmp21 = tmp18.to(tl.float32)
    tmp22 = tl.full([1, 1], False, tl.int1)
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp23, tmp21)
    tl.atomic_add(out_ptr1 + (x5 + (6*tmp20)), tmp24, None)
''')
