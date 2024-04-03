

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/ik/cikbpbemox3w5xj5letap373u5l6my4w5yv2moqfssbmdnoxsi2v.py
# Source Nodes: [add_12, l__self___distilbert_transformer_layer_5_ffn_dropout, l__self___distilbert_transformer_layer_5_output_layer_norm, l__self___distilbert_transformer_layer_5_sa_layer_norm, l__self___dropout, l__self___qa_outputs], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_12 => add_42
# l__self___distilbert_transformer_layer_5_ffn_dropout => gt_12, mul_66, mul_67
# l__self___distilbert_transformer_layer_5_output_layer_norm => add_43, add_44, mul_68, mul_69, rsqrt_12, sub_18, var_mean_12
# l__self___distilbert_transformer_layer_5_sa_layer_norm => add_40, mul_62
# l__self___dropout => gt_13, mul_70, mul_71
# l__self___qa_outputs => convert_element_type_126, view_138
triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_16', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*i1', 9: '*fp32', 10: '*bf16', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr6, out_ptr7, out_ptr8, out_ptr9, load_seed_offset, load_seed_offset1, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tl.full([1], 768, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 / tmp26
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp34 = tl.load(in_ptr0 + load_seed_offset1)
    tmp35 = tl.rand(tmp34, (tmp1).to(tl.uint32))
    tmp36 = tmp35 > tmp4
    tmp37 = tmp17 - tmp27
    tmp38 = 768.0
    tmp39 = tmp33 / tmp38
    tmp40 = 1e-12
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp44 = tmp36.to(tl.float32)
    tmp46 = tmp43 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp44 * tmp48
    tmp50 = tmp49 * tmp9
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp42 / tmp38
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp36, rmask)
    tl.store(out_ptr7 + (r1 + (768*x0)), tmp43, rmask)
    tl.store(out_ptr8 + (r1 + (768*x0)), tmp51, rmask)
    tl.store(out_ptr9 + (x0), tmp52, None)
''')
