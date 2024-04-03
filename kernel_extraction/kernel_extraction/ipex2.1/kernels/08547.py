

# Original file: ./DistillGPT2__0_backward_99.1/DistillGPT2__0_backward_99.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/zx/czxegoqaxivkyyrm5aulbgoj5tqhkn7urxdtuecv2dydcsm4aznj.py
# Source Nodes: [cross_entropy, l__self___transformer_h_0_ln_1], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_13
# l__self___transformer_h_0_ln_1 => mul_2, sub
triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28 = async_compile.triton('triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28', '''
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
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*i64', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_native_layer_norm_backward_nll_loss_forward_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask)
    tmp32 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 768.0
    tmp20 = tmp11 / tmp19
    tmp21 = tmp3 * tmp19
    tmp22 = tmp21 - tmp7
    tmp23 = tmp12 * tmp17
    tmp24 = tmp22 - tmp23
    tmp25 = tmp20 * tmp24
    tmp26 = tmp18 + tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.1111111111111112
    tmp30 = tmp28 * tmp29
    tmp31 = tmp26 * tmp30
    tmp33 = tl.where(tmp32 < 0, tmp32 + 50257, tmp32)
    tmp34 = tl.full([1], -1, tl.int64)
    tmp35 = tmp32 == tmp34
    tmp36 = 0.0
    tmp37 = tl.where(tmp35, tmp36, tmp31)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp31, rmask)
    tl.atomic_add(out_ptr2 + (tl.broadcast_to(r1 + (768*tmp33), [RBLOCK])), tmp37, rmask)
''')
