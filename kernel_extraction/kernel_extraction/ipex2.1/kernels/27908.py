

# Original file: ./BertForMaskedLM__0_forward_205.0/BertForMaskedLM__0_forward_205.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/xu/cxucxgnkr7mstuq66ajyqhuqrwk5vbuzbmor7uy2jlpxneuy6bfy.py
# Source Nodes: [add_2, l__mod___bert_embeddings_dropout, l__mod___bert_embeddings_layer_norm, l__mod___bert_encoder_layer_0_attention_output_dropout, l__mod___bert_encoder_layer_0_attention_output_layer_norm, l__mod___bert_encoder_layer_0_intermediate_dense], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_2 => add_5
# l__mod___bert_embeddings_dropout => mul_3, mul_4
# l__mod___bert_embeddings_layer_norm => add_3, mul_2
# l__mod___bert_encoder_layer_0_attention_output_dropout => gt_2, mul_7, mul_8
# l__mod___bert_encoder_layer_0_attention_output_layer_norm => add_6, add_7, mul_10, mul_9, rsqrt_1, sub_3, var_mean_1
# l__mod___bert_encoder_layer_0_intermediate_dense => view_18
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
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
    tmp6 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask)
    tmp12 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tmp17 * tmp8
    tmp19 = tmp9 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 768, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tmp19 - tmp29
    tmp37 = 768.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-12
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp41 / tmp37
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp42, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp46, rmask)
    tl.store(out_ptr6 + (x0), tmp47, None)
''')
