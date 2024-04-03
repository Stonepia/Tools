

# Original file: ./RobertaForCausalLM__0_forward_133.0/RobertaForCausalLM__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/kk/ckkj5sjv5qokdndabklhzembk7nug4o4t6bek3buijdyyvawyzpf.py
# Source Nodes: [add_4, l__self___roberta_embeddings_dropout, l__self___roberta_embeddings_layer_norm, l__self___roberta_encoder_layer_0_attention_output_dropout, l__self___roberta_encoder_layer_0_attention_output_layer_norm, l__self___roberta_encoder_layer_0_intermediate_dense], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_4 => add_7
# l__self___roberta_embeddings_dropout => mul_4, mul_5
# l__self___roberta_embeddings_layer_norm => add_5, mul_3
# l__self___roberta_encoder_layer_0_attention_output_dropout => gt_2, mul_8, mul_9
# l__self___roberta_encoder_layer_0_attention_output_layer_norm => add_8, add_9, mul_10, mul_11, rsqrt_1, sub_3, var_mean_1
# l__self___roberta_encoder_layer_0_intermediate_dense => convert_element_type_15, view_18
triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5', '''
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
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp16', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask)
    tmp14 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = tmp13 * tmp18
    tmp20 = tmp19 * tmp9
    tmp21 = tmp11 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tl.full([1], 768, tl.int32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = tmp21 - tmp31
    tmp39 = 768.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-12
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp43 / tmp39
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp44, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp49, rmask)
    tl.store(out_ptr7 + (x0), tmp50, None)
''')
