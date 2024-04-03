

# Original file: ./volo_d1_224___60.0/volo_d1_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/wd/cwdhav5ncnaguaf7zchb6ijc46pen4mhxwpegsraotbsanhx7bgw.py
# Source Nodes: [add_8, add_9, getattr_l__self___network_2___0___attn_proj_drop, getattr_l__self___network_2___0___mlp_fc1, getattr_l__self___network_2___0___norm2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_8 => add_54
# add_9 => add_57
# getattr_l__self___network_2___0___attn_proj_drop => clone_51
# getattr_l__self___network_2___0___mlp_fc1 => convert_element_type_107
# getattr_l__self___network_2___0___norm2 => add_58, add_59, clone_52, mul_44, mul_45, rsqrt_9, sub_17, var_mean_9
triton_per_fused__to_copy_add_clone_native_layer_norm_23 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_23', '''
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
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 12544
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 196
    tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 384, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 384.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp34, rmask & xmask)
''')
