

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/w3/cw3r7ekdhed3aluwr7gbq22buptloujio3jknep5bo6n7hmbqtib.py
# Source Nodes: [add_5, add_6, add_7, add_8, getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_proj_drop, getattr_getattr_l__self___levels___1___transformer_encoder___0___mlp_drop2, getattr_getattr_l__self___levels___1___transformer_encoder___1___attn_proj_drop, getattr_getattr_l__self___levels___1___transformer_encoder___1___mlp_fc1, layer_norm_8], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_5 => add_17
# add_6 => add_20
# add_7 => add_24
# add_8 => add_27
# getattr_getattr_l__self___levels___1___transformer_encoder___0___attn_proj_drop => clone_21
# getattr_getattr_l__self___levels___1___transformer_encoder___0___mlp_drop2 => clone_23
# getattr_getattr_l__self___levels___1___transformer_encoder___1___attn_proj_drop => clone_28
# getattr_getattr_l__self___levels___1___transformer_encoder___1___mlp_fc1 => convert_element_type_57
# layer_norm_8 => add_28, add_29, mul_33, mul_34, rsqrt_8, sub_12, var_mean_8
triton_per_fused__to_copy_add_clone_native_layer_norm_24 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_24', '''
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
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 25088
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    x2 = (xindex // 784)
    x4 = xindex % 784
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 % 2)) + (28*(x0 // 14)) + (392*(x1 // 2)) + (784*r3) + (200704*x2) + (x0 % 14)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r3 + (256*x5)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r3 + (256*x5)), rmask & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r3 + (256*x5)), rmask & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
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
    tmp20 = tl.full([1], 256, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 256.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r3 + (256*x5)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (256*x5)), tmp40, rmask & xmask)
''')
