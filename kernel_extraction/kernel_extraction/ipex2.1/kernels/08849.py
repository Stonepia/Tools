

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/yq/cyq5eplfsvb3hkj6xlnz3tmsj4m6r4frbp6mkmfaj7yeashpiyws.py
# Source Nodes: [add_10, add_11, add_12, add_13, getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_proj_drop, getattr_getattr_l__mod___levels___2___transformer_encoder___0___mlp_drop2, getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_proj_drop, layer_norm_13], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_10 => add_34
# add_11 => add_37
# add_12 => add_41
# add_13 => add_44
# getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_proj_drop => clone_36
# getattr_getattr_l__mod___levels___2___transformer_encoder___0___mlp_drop2 => clone_38
# getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_proj_drop => clone_43
# layer_norm_13 => add_45, add_46, convert_element_type_48, convert_element_type_49, mul_53, mul_54, rsqrt_13, sub_19, var_mean_13
triton_per_fused_add_clone_native_layer_norm_38 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_38', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_38', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (100352*x1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r2 + (512*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (r2 + (512*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tl.full([1], 512, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp9 - tmp19
    tmp27 = 512.0
    tmp28 = tmp25 / tmp27
    tmp29 = 1e-06
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r2 + (512*x3)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp39, rmask & xmask)
''')
