

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/qu/cqun7ljs4aqmnhk6ka7xm2i2zu3vsfxgylmgl7zi4n5doub7o7p6.py
# Source Nodes: [add_4, l__self___transformer_dropout, l__self___transformer_layer_0_ff_layer_1, l__self___transformer_layer_0_rel_attn_dropout_1, l__self___transformer_layer_0_rel_attn_layer_norm, l__self___transformer_word_embedding], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# add_4 => add_4
# l__self___transformer_dropout => mul, mul_1
# l__self___transformer_layer_0_ff_layer_1 => convert_element_type_16, view_34
# l__self___transformer_layer_0_rel_attn_dropout_1 => gt_3, mul_10, mul_9
# l__self___transformer_layer_0_rel_attn_layer_norm => add_5, add_6, mul_11, mul_12, rsqrt, sub_1, var_mean
# l__self___transformer_word_embedding => embedding
triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.triton('triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_10', '''
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
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*i1', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp16', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_native_dropout_native_layer_norm_native_layer_norm_backward_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr5, out_ptr6, out_ptr7, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
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
    tmp7 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tl.where(tmp14 < 0, tmp14 + 32000, tmp14)
    # tl.device_assert((0 <= tmp15) & (tmp15 < 32000), "index out of bounds: 0 <= tmp15 < 32000")
    tmp16 = tl.load(in_ptr4 + (r1 + (1024*tmp15)), rmask, other=0.0)
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 * tmp9
    tmp19 = tmp11 + tmp18
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
    tmp36 = tmp19 - tmp29
    tmp37 = 1024.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-12
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp41 / tmp37
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp5, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp42, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp47, rmask)
    tl.store(out_ptr7 + (x0), tmp48, None)
''')
