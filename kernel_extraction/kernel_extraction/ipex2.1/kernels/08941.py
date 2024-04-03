

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/76/c76izqqg66dr6u54u34is3rlnp7r6x2m6d45qfg4clvcua2p44oe.py
# Source Nodes: [add_14, add_15, add_16, add_17, getattr_getattr_l__self___levels___2___transformer_encoder___1___mlp_drop2, getattr_getattr_l__self___levels___2___transformer_encoder___2___attn_proj_drop, getattr_getattr_l__self___levels___2___transformer_encoder___2___mlp_drop2, getattr_getattr_l__self___levels___2___transformer_encoder___3___attn_proj_drop, getattr_getattr_l__self___levels___2___transformer_encoder___3___mlp_fc1, layer_norm_17], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_14 => add_48
# add_15 => add_51
# add_16 => add_55
# add_17 => add_58
# getattr_getattr_l__self___levels___2___transformer_encoder___1___mlp_drop2 => clone_45
# getattr_getattr_l__self___levels___2___transformer_encoder___2___attn_proj_drop => clone_50
# getattr_getattr_l__self___levels___2___transformer_encoder___2___mlp_drop2 => clone_52
# getattr_getattr_l__self___levels___2___transformer_encoder___3___attn_proj_drop => clone_57
# getattr_getattr_l__self___levels___2___transformer_encoder___3___mlp_fc1 => convert_element_type_118
# layer_norm_17 => add_59, add_60, mul_71, mul_72, rsqrt_17, sub_25, var_mean_17
triton_per_fused__to_copy_add_clone_native_layer_norm_42 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_42', '''
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
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1 + (512*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 512, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 512.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp40, rmask & xmask)
''')
