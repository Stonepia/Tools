

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/s7/cs7gcqbeqxddlvrtowgm5xvodfbp3q4d5s4rnrwkpvtii3i7epjn.py
# Source Nodes: [add_8, l__self___layer_0_output_layer_norm, l__self___layer_1_attention_output_dense, l__self___layer_1_attention_output_dropout, l__self___layer_1_attention_output_layer_norm, l__self___layer_1_intermediate_dense], Original ATen: [aten._to_copy, aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_8 => add_23
# l__self___layer_0_output_layer_norm => add_14, mul_13
# l__self___layer_1_attention_output_dense => add_22, convert_element_type_42
# l__self___layer_1_attention_output_dropout => gt_4, mul_17, mul_18
# l__self___layer_1_attention_output_layer_norm => add_24, add_25, mul_19, mul_20, rsqrt_2, sub_14, var_mean_2
# l__self___layer_1_intermediate_dense => convert_element_type_43, view_150
triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_34 = async_compile.triton('triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_34', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*fp16', 11: '*fp32', 12: 'i32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_dropout_native_layer_norm_native_layer_norm_backward_view_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
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
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp14 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 768.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp42 / tmp38
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp43, rmask)
    tl.store(out_ptr6 + (r1 + (768*x0)), tmp48, rmask)
    tl.store(out_ptr7 + (x0), tmp49, None)
''')
