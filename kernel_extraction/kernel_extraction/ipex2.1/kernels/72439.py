

# Original file: ./YituTechConvBert__0_backward_243.1/YituTechConvBert__0_backward_243.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ff/cffc77pls4rxljnqj6znqecc7fjhb4chkiibkdfqwlhlcmdtzww4.py
# Source Nodes: [cross_entropy], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_14
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_33 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_33', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*i64', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r2) + (393216*x1)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask)
    tmp16 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr7 + (r2 + (768*x3)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp10 * tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp36 = tl.where(tmp35 < 0, tmp35 + 2, tmp35)
    tmp37 = tl.full([1], -1, tl.int64)
    tmp38 = tmp35 == tmp37
    tmp39 = 0.0
    tmp40 = tl.where(tmp38, tmp39, tmp34)
    tmp42 = tl.where(tmp41 < 0, tmp41 + 30522, tmp41)
    tmp43 = tl.full([1], 0, tl.int64)
    tmp44 = tmp41 == tmp43
    tmp45 = tl.where(tmp44, tmp39, tmp34)
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp15, rmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp34, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r2 + (768*tmp36), [RBLOCK])), tmp40, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r2 + (768*tmp42), [RBLOCK])), tmp45, rmask)
''')
