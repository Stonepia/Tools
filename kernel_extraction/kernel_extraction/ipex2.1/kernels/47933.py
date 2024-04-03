

# Original file: ./MegatronBertForCausalLM__0_backward_279.1/MegatronBertForCausalLM__0_backward_279.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/yv/cyvwt4vvghhplxttjhxgncyipdec3cj3snlbgtsaozvs2aqvsza7.py
# Source Nodes: [cross_entropy, l__self___bert_encoder_layer_0_attention_ln], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_3
# l__self___bert_encoder_layer_0_attention_ln => mul_3, sub_1
triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_31 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_31', '''
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i64', 8: '*fp32', 9: '*i1', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp35 = tl.load(in_ptr9 + (r1 + (1024*x0)), rmask)
    tmp43 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = tmp14 - tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp9 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = 1024.0
    tmp25 = tmp9 * tmp24
    tmp26 = tmp25 - tmp13
    tmp27 = tmp18 * tmp23
    tmp28 = tmp26 - tmp27
    tmp30 = tl.where(tmp29 < 0, tmp29 + 2, tmp29)
    tmp32 = tmp17 / tmp24
    tmp33 = tmp32 * tmp28
    tmp34 = tmp31 + tmp33
    tmp36 = tmp35.to(tl.float32)
    tmp37 = 1.1111111111111112
    tmp38 = tmp36 * tmp37
    tmp39 = tmp34 * tmp38
    tmp40 = tl.full([1], False, tl.int1)
    tmp41 = 0.0
    tmp42 = tl.where(tmp40, tmp41, tmp39)
    tmp44 = tl.where(tmp43 < 0, tmp43 + 29056, tmp43)
    tmp45 = tl.full([1], 0, tl.int64)
    tmp46 = tmp43 == tmp45
    tmp47 = tl.where(tmp46, tmp41, tmp39)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp28, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (1024*tmp30), [RBLOCK])), tmp42, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (1024*tmp44), [RBLOCK])), tmp47, rmask)
''')
