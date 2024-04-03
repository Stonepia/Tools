

# Original file: ./RobertaForCausalLM__0_backward_135.1/RobertaForCausalLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/yj/cyjnb3xchr7huwxmp5r4hujqywq3ewfa44oeuoj3qfl6latyndnz.py
# Source Nodes: [cross_entropy], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_27 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_27', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4', 'out_ptr5'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
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
    x2 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp35 = tl.where(tmp34 < 0, tmp34 + 512, tmp34)
    tmp36 = tl.full([1], 0, tl.int64)
    tmp37 = tmp34 == tmp36
    tmp38 = 0.0
    tmp39 = tl.where(tmp37, tmp38, tmp33)
    tmp41 = tl.where(tmp40 < 0, tmp40 + 2, tmp40)
    tmp42 = tl.full([1], -1, tl.int64)
    tmp43 = tmp40 == tmp42
    tmp44 = tl.where(tmp43, tmp38, tmp33)
    tmp46 = tl.where(tmp45 < 0, tmp45 + 50265, tmp45)
    tmp47 = tmp45 == tmp36
    tmp48 = tl.where(tmp47, tmp38, tmp33)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp14, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (768*tmp35), [RBLOCK])), tmp39, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (768*tmp41), [RBLOCK])), tmp44, rmask)
    tl.atomic_add(out_ptr5 + (tl.broadcast_to(r1 + (768*tmp46), [RBLOCK])), tmp48, rmask)
''')
