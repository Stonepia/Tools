

# Original file: ./YituTechConvBert__0_backward_207.1/YituTechConvBert__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/d5/cd5orbseiqdyalizryjzlkzk42yi3fim6m6oua7kmnxj4itz5j5l.py
# Source Nodes: [cross_entropy], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_14
triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_43 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_43', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*i64', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_43', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + (512*r2) + (393216*x1)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask)
    tmp21 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr7 + (r2 + (768*x3)), rmask, other=0.0)
    tmp33 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 1.1111111111111112
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 * tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tmp22 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = 768.0
    tmp35 = tmp22 * tmp34
    tmp36 = tmp35 - tmp26
    tmp37 = tmp27 * tmp32
    tmp38 = tmp36 - tmp37
    tmp39 = tmp33 * tmp38
    tmp41 = tl.where(tmp40 < 0, tmp40 + 2, tmp40)
    tmp42 = tl.full([1], -1, tl.int64)
    tmp43 = tmp40 == tmp42
    tmp44 = 0.0
    tmp45 = tl.where(tmp43, tmp44, tmp39)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 30522, tmp46)
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp46 == tmp48
    tmp50 = tl.where(tmp49, tmp44, tmp39)
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp20, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r2 + (768*tmp41), [RBLOCK])), tmp45, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r2 + (768*tmp47), [RBLOCK])), tmp50, rmask)
''')
