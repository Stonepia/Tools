

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/ro/cro5nec3gpkkftgmpzgs3mvd7nxfo7yrbki5lrhenelqqhfgtkuu.py
# Source Nodes: [add_12, l__mod___distilbert_transformer_layer_5_ffn_dropout, l__mod___distilbert_transformer_layer_5_output_layer_norm, l__mod___distilbert_transformer_layer_5_sa_layer_norm, l__mod___dropout, l__mod___qa_outputs], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_12 => add_42
# l__mod___distilbert_transformer_layer_5_ffn_dropout => gt_12, mul_66, mul_67
# l__mod___distilbert_transformer_layer_5_output_layer_norm => add_43, add_44, mul_68, mul_69, rsqrt_12, sub_18, var_mean_12
# l__mod___distilbert_transformer_layer_5_sa_layer_norm => add_40, mul_62
# l__mod___dropout => gt_13, mul_70, mul_71
# l__mod___qa_outputs => view_138
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_12 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_12', '''
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
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr5, out_ptr6, out_ptr7, out_ptr8, load_seed_offset, load_seed_offset1, xnumel, rnumel):
    xnumel = 32768
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
    tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp9 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = tl.load(in_ptr0 + load_seed_offset1)
    tmp33 = tl.rand(tmp32, (tmp1).to(tl.uint32))
    tmp34 = tmp33 > tmp3
    tmp35 = tmp15 - tmp25
    tmp36 = 768.0
    tmp37 = tmp31 / tmp36
    tmp38 = 1e-12
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp42 = tmp34.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp42 * tmp46
    tmp48 = tmp47 * tmp8
    tmp49 = tmp40 / tmp36
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp34, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp41, rmask)
    tl.store(out_ptr7 + (r1 + (768*x0)), tmp48, rmask)
    tl.store(out_ptr8 + (x0), tmp49, None)
''')
