

# Original file: ./MegatronBertForCausalLM__0_forward_205.0/MegatronBertForCausalLM__0_forward_205.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/fr/cfralyyv2tktuyzuhudot5zqhh2dzvpvmk56wo7p5dnldli5g75w.py
# Source Nodes: [add_2, add_3, add_5, l__self___bert_encoder_layer_1_attention_output_dropout, l__self___bert_encoder_layer_1_intermediate_dense, l__self___bert_encoder_layer_1_ln], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.view]
# add_2 => add_5
# add_3 => add_9
# add_5 => add_13
# l__self___bert_encoder_layer_1_attention_output_dropout => gt_5, mul_20, mul_21
# l__self___bert_encoder_layer_1_intermediate_dense => convert_element_type_31, view_40
# l__self___bert_encoder_layer_1_ln => add_14, add_15, mul_22, mul_23, rsqrt_3, sub_6, var_mean_3
triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_10 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_10', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_native_layer_norm_view_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
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
    tmp43 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp18 = tmp10.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 1024, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = 1024.0
    tmp37 = tmp35 / tmp36
    tmp38 = 1e-12
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp19 - tmp29
    tmp42 = tmp41 * tmp40
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp40, None)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp47, rmask)
    tl.store(out_ptr2 + (x0), tmp29, None)
''')
