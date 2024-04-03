

# Original file: ./DistillGPT2__0_forward_97.0/DistillGPT2__0_forward_97.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/y6/cy6a2f3pd34g2sz6qdcioao566jae54vhz3k2xpkzm6glwybpxeg.py
# Source Nodes: [add_4, add_5, l__mod___transformer_h_0_mlp_dropout, l__mod___transformer_h_1_attn_resid_dropout, l__mod___transformer_h_1_ln_2], Original ATen: [aten.add, aten.native_dropout, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_4 => add_8
# add_5 => add_11
# l__mod___transformer_h_0_mlp_dropout => mul_14, mul_15
# l__mod___transformer_h_1_attn_resid_dropout => gt_5, mul_20, mul_21
# l__mod___transformer_h_1_ln_2 => add_12, add_13, mul_22, mul_23, rsqrt_3, sub_5, var_mean_3
triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10', '''
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
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, rnumel):
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
    tmp10 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask)
    tmp13 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0.0)
    tmp41 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp10 + tmp15
    tmp17 = tmp9 + tmp16
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
    tmp34 = tmp17 - tmp27
    tmp35 = 768.0
    tmp36 = tmp33 / tmp35
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp34 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tmp39 / tmp35
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp4, rmask)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp17, rmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp40, rmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp44, rmask)
    tl.store(out_ptr6 + (x0), tmp45, None)
''')
