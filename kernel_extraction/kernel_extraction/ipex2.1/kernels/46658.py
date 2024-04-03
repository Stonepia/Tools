

# Original file: ./DistilBertForMaskedLM__0_backward_99.1/DistilBertForMaskedLM__0_backward_99.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/zi/czityvehp72evwipwbo3qp5djf2gjmljrkm7kuksyo7mb5j7xgsp.py
# Source Nodes: [add, l__mod___distilbert_embeddings_layer_norm, l__mod___mlm_loss_fct], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward]
# add => add
# l__mod___distilbert_embeddings_layer_norm => mul, sub
# l__mod___mlm_loss_fct => full_default_7
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_24 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_24', '''
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16384
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 - tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp13 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 768.0
    tmp31 = tmp23 / tmp30
    tmp32 = tmp13 * tmp30
    tmp33 = tmp32 - tmp17
    tmp34 = tmp24 * tmp29
    tmp35 = tmp33 - tmp34
    tmp36 = tmp31 * tmp35
    tmp38 = tl.where(tmp37 < 0, tmp37 + 30522, tmp37)
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp37 == tmp39
    tmp41 = 0.0
    tmp42 = tl.where(tmp40, tmp41, tmp36)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp36, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (768*tmp38), [RBLOCK])), tmp42, rmask)
''')
