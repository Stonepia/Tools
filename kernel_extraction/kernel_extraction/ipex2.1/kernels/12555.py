

# Original file: ./MegatronBertForCausalLM__0_forward_205.0/MegatronBertForCausalLM__0_forward_205.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/fy/cfyhtwb3envvg5ghvjozlxv7c5l2pwaofhd3fgt5367q67iph3lr.py
# Source Nodes: [add_2, add_3, add_5, add_6, l__self___bert_encoder_layer_1_output_dropout, l__self___bert_encoder_layer_2_attention_ln, l__self___bert_encoder_layer_2_attention_self_query], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_2 => add_5
# add_3 => add_9
# add_5 => add_13
# add_6 => add_17
# l__self___bert_encoder_layer_1_output_dropout => gt_6, mul_27, mul_28
# l__self___bert_encoder_layer_2_attention_ln => add_18, add_19, mul_29, mul_30, rsqrt_4, sub_7, var_mean_4
# l__self___bert_encoder_layer_2_attention_self_query => convert_element_type_38, view_44
triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_11 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: 'i32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp46 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (1024*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp10.to(tl.float32)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp28 = tl.where(rmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tl.full([1], 1024, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp39 = 1024.0
    tmp40 = tmp38 / tmp39
    tmp41 = 1e-12
    tmp42 = tmp40 + tmp41
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp22 - tmp32
    tmp45 = tmp44 * tmp43
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tmp49.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp10, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp22, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp43, None)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp50, rmask)
    tl.store(out_ptr3 + (x0), tmp32, None)
''')
