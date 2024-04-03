

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/7l/c7lajczwnbsmab2gx24q7rhrnj3dj6pw7hii3a56xwrxsqpndtm7.py
# Source Nodes: [add_12, l__mod___distilbert_transformer_layer_5_ffn_dropout, l__mod___distilbert_transformer_layer_5_output_layer_norm, l__mod___dropout, l__mod___qa_outputs], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_12 => add_42
# l__mod___distilbert_transformer_layer_5_ffn_dropout => gt_12, mul_66, mul_67
# l__mod___distilbert_transformer_layer_5_output_layer_norm => add_43, add_44, convert_element_type_54, convert_element_type_55, mul_68, mul_69, rsqrt_12, sub_18, var_mean_12
# l__mod___dropout => gt_13, mul_70, mul_71
# l__mod___qa_outputs => view_138
triton_per_fused_add_native_dropout_native_layer_norm_view_11 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_view_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*i1', 7: '*fp32', 8: '*i1', 9: '*fp16', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_view_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr4, out_ptr5, load_seed_offset, load_seed_offset1, xnumel, rnumel):
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 768, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = 768.0
    tmp31 = tmp29 / tmp30
    tmp32 = 1e-12
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tl.load(in_ptr0 + load_seed_offset1)
    tmp36 = tl.rand(tmp35, (tmp1).to(tl.uint32))
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 > tmp4
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp13 - tmp23
    tmp41 = tmp40 * tmp34
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 * tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp39 * tmp48
    tmp50 = tmp49 * tmp9
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp34, None)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp38, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp50, rmask)
    tl.store(out_ptr2 + (x0), tmp23, None)
''')
