

# Original file: ./LayoutLMForSequenceClassification__0_backward_171.1/LayoutLMForSequenceClassification__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/ed/cedwmqmelff7u3po2uivpe5hz2ohuazi6tkg2jtn6tbiqn6qrqr2.py
# Source Nodes: [cross_entropy], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_5
triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_30 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_30', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*i64', 11: '*i64', 12: '*i64', 13: '*i64', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32', 23: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4', 'out_ptr5', 'out_ptr6', 'out_ptr7', 'out_ptr8', 'out_ptr9'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (4*x0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr9 + (4*x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr10 + (4*x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr11 + (4*x0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
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
    tmp22 = tmp16 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = 768.0
    tmp29 = tmp16 * tmp28
    tmp30 = tmp29 - tmp20
    tmp31 = tmp21 * tmp26
    tmp32 = tmp30 - tmp31
    tmp33 = tmp27 * tmp32
    tmp35 = tl.where(tmp34 < 0, tmp34 + 2, tmp34)
    tmp36 = tl.full([1], False, tl.int1)
    tmp37 = 0.0
    tmp38 = tl.where(tmp36, tmp37, tmp33)
    tmp39 = tl.where(tmp34 < 0, tmp34 + 1024, tmp34)
    tmp41 = tl.where(tmp40 < 0, tmp40 + 1024, tmp40)
    tmp43 = tl.where(tmp42 < 0, tmp42 + 1024, tmp42)
    tmp45 = tl.where(tmp44 < 0, tmp44 + 1024, tmp44)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 1024, tmp46)
    tmp49 = tl.where(tmp48 < 0, tmp48 + 30522, tmp48)
    tmp50 = tl.full([1], 0, tl.int64)
    tmp51 = tmp48 == tmp50
    tmp52 = tl.where(tmp51, tmp37, tmp33)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp14, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp33, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (768*tmp35), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp39), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r1 + (768*tmp41), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr6 + (tl.broadcast_to(r1 + (768*tmp43), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr7 + (tl.broadcast_to(r1 + (768*tmp45), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr8 + (tl.broadcast_to(r1 + (768*tmp47), [RBLOCK])), tmp38, rmask)
    tl.atomic_add(out_ptr9 + (tl.broadcast_to(r1 + (768*tmp49), [RBLOCK])), tmp52, rmask)
''')
