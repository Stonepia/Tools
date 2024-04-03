

# Original file: ./deit_base_distilled_patch16_224___60.0/deit_base_distilled_patch16_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/mm/cmmaezskzbmrkt76agtz42dt4vokcryztlrcijdkfr4oabig7l7m.py
# Source Nodes: [add, add_1, add_2, add_3, getattr_l__self___blocks___0___attn_proj_drop, getattr_l__self___blocks___0___mlp_drop2, getattr_l__self___blocks___1___attn_proj_drop, getattr_l__self___blocks___1___mlp_fc1, getattr_l__self___blocks___1___norm2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add => add
# add_1 => add_3
# add_2 => add_7
# add_3 => add_10
# getattr_l__self___blocks___0___attn_proj_drop => clone_1
# getattr_l__self___blocks___0___mlp_drop2 => clone_3
# getattr_l__self___blocks___1___attn_proj_drop => clone_4
# getattr_l__self___blocks___1___mlp_fc1 => convert_element_type_21
# getattr_l__self___blocks___1___norm2 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_3, var_mean_3
triton_per_fused__to_copy_add_clone_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_clone_native_layer_norm_10', '''
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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 12672
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 198
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (768*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tmp11 - tmp21
    tmp29 = 768.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-06
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp39, rmask & xmask)
''')
