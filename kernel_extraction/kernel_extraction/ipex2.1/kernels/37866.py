

# Original file: ./ElectraForQuestionAnswering__0_backward_135.1/ElectraForQuestionAnswering__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/qq/cqqnjksq2ddmvm4tpstyode5lahfayiuft74zwedjzy2yjliajk5.py
# Source Nodes: [cross_entropy], Original ATen: [aten._to_copy, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_per_fused__to_copy_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_25 = async_compile.triton('triton_per_fused__to_copy_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_25', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask)
    tmp7 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 128.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tl.where(tmp26 < 0, tmp26 + 2, tmp26)
    tmp28 = tl.full([1, 1], -1, tl.int64)
    tmp29 = tmp26 == tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp29, tmp30, tmp25)
    tmp33 = tl.where(tmp32 < 0, tmp32 + 30522, tmp32)
    tmp34 = tl.full([1, 1], 0, tl.int64)
    tmp35 = tmp32 == tmp34
    tmp36 = tl.where(tmp35, tmp30, tmp25)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp25, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (128*tmp27), [XBLOCK, RBLOCK])), tmp31, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (128*tmp33), [XBLOCK, RBLOCK])), tmp36, rmask)
''')
