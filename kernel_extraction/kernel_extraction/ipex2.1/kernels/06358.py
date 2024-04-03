

# Original file: ./ElectraForCausalLM__0_backward_207.1/ElectraForCausalLM__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/nh/cnh5nt5jhhooqkheotru77p2ema4tg65c4re3bf5225qun4b4b3k.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_28 = async_compile.triton('triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_28', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp6 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp7 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp19 = 128.0
    tmp20 = tmp7 * tmp19
    tmp21 = tmp20 - tmp11
    tmp22 = tmp12 * tmp17
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp26 = tl.where(tmp25 < 0, tmp25 + 2, tmp25)
    tmp27 = tl.full([1, 1], -1, tl.int64)
    tmp28 = tmp25 == tmp27
    tmp29 = 0.0
    tmp30 = tl.where(tmp28, tmp29, tmp24)
    tmp32 = tl.where(tmp31 < 0, tmp31 + 30522, tmp31)
    tmp33 = tl.full([1, 1], 0, tl.int64)
    tmp34 = tmp31 == tmp33
    tmp35 = tl.where(tmp34, tmp29, tmp24)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp24, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (128*tmp26), [XBLOCK, RBLOCK])), tmp30, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (128*tmp32), [XBLOCK, RBLOCK])), tmp35, rmask)
''')
