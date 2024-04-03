

# Original file: ./hf_GPT2___60.0/hf_GPT2___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/pn/cpnigjp3k4r33b5cztt6vqppaksom4m557qffja7gmbh7uppehzq.py
# Source Nodes: [add_5, add_8, add_9, addmm_10, l__self___transformer_h_1_attn_resid_dropout, l__self___transformer_h_1_mlp_dropout, l__self___transformer_h_2_attn_resid_dropout, l__self___transformer_h_2_ln_2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_5 => add_11
# add_8 => add_16
# add_9 => add_19
# addmm_10 => convert_element_type_36
# l__self___transformer_h_1_attn_resid_dropout => clone_7
# l__self___transformer_h_1_mlp_dropout => clone_8
# l__self___transformer_h_2_attn_resid_dropout => clone_11
# l__self___transformer_h_2_ln_2 => var_mean_5
triton_per_fused__to_copy_add_clone_native_layer_norm_11 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_11', '''
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
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp9 = tmp1 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 768.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp37, rmask & xmask)
''')
