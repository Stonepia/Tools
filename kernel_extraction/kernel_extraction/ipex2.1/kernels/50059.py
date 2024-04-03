

# Original file: ./MegatronBertForQuestionAnswering__0_backward_207.1/MegatronBertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/5y/c5yag74pxaj6l6te27sbhdyrzl6wwlw5mx6uiglkjas374vcyyp7.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.sum]

triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_24 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_24', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*i64', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*r2)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (x3 + (524288*r2)), rmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x3 + (524288*r2)), rmask)
    tmp17 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1024.0
    tmp3 = tmp1 / tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp7 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = tl.where(tmp17 < 0, tmp17 + 512, tmp17)
    tmp19 = tl.full([1, 1], -1, tl.int64)
    tmp20 = tmp17 == tmp19
    tmp21 = tmp16.to(tl.float32)
    tmp22 = 0.0
    tmp23 = tl.where(tmp20, tmp22, tmp21)
    tl.atomic_add(out_ptr1 + (x0 + (1024*tmp18)), tmp23, None)
''')
