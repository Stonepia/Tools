

# Original file: ./DistilBertForMaskedLM__0_backward_99.1/DistilBertForMaskedLM__0_backward_99.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/ff/cff3iemyi2zkmyuky2uqna633hyh7kdg3iz7oa6a27qtbn6245fh.py
# Source Nodes: [add, l__self___distilbert_embeddings_layer_norm, l__self___mlm_loss_fct], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward]
# add => add
# l__self___distilbert_embeddings_layer_norm => mul, sub
# l__self___mlm_loss_fct => full_default_7
triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp9 * tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 - tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tmp16 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = 768.0
    tmp34 = tmp26 / tmp33
    tmp35 = tmp16 * tmp33
    tmp36 = tmp35 - tmp20
    tmp37 = tmp27 * tmp32
    tmp38 = tmp36 - tmp37
    tmp39 = tmp34 * tmp38
    tmp41 = tl.where(tmp40 < 0, tmp40 + 30522, tmp40)
    tmp42 = tl.full([1], 0, tl.int64)
    tmp43 = tmp40 == tmp42
    tmp44 = 0.0
    tmp45 = tl.where(tmp43, tmp44, tmp39)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp14, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp39, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (768*tmp41), [RBLOCK])), tmp45, rmask)
''')
