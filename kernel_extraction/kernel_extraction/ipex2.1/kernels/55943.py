

# Original file: ./cait_m36_384___60.0/cait_m36_384___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/2w/c2wuqxywwsi6l2j4xa4opd5sftkkygr5gbewvvo2mgebawy5rktr.py
# Source Nodes: [add, add_1, add_2, getattr_l__self___blocks___0___attn_proj_drop, getattr_l__self___blocks___0___mlp_drop2, getattr_l__self___blocks___1___attn_qkv, getattr_l__self___blocks___1___norm1, mul_1, mul_2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mul, aten.native_layer_norm]
# add => add
# add_1 => add_5
# add_2 => add_9
# getattr_l__self___blocks___0___attn_proj_drop => clone_5
# getattr_l__self___blocks___0___mlp_drop2 => clone_8
# getattr_l__self___blocks___1___attn_qkv => convert_element_type_21
# getattr_l__self___blocks___1___norm1 => add_10, add_11, clone_9, mul_10, mul_11, rsqrt_2, sub_3, var_mean_2
# mul_1 => mul_3
# mul_2 => mul_9
triton_per_fused__to_copy_add_clone_mul_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_clone_mul_native_layer_norm_10', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*bf16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mul_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mul_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 576
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
    tmp2 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp8 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 768, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp30 = tmp13 - tmp23
    tmp31 = 768.0
    tmp32 = tmp29 / tmp31
    tmp33 = 1e-06
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp38 = tmp36 * tmp37
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp13, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp41, rmask & xmask)
''')
