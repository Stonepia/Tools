

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/cu/ccu5kurlbcewgxiduhwk72heavwxk7b2jrlyzbjawcfwofi6pib6.py
# Source Nodes: [add_2, add_3, l__self___blocks_0_mlp_in_drop2, l__self___blocks_0_norm1_proj, l__self___blocks_0_proj, l__self___blocks_1_attn_in_qk, l__self___blocks_1_norm_in], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_2 => add_10
# add_3 => add_14
# l__self___blocks_0_mlp_in_drop2 => clone_11
# l__self___blocks_0_norm1_proj => add_15, clone_12, mul_12, rsqrt_4, sub_5, var_mean_4
# l__self___blocks_0_proj => convert_element_type_26
# l__self___blocks_1_attn_in_qk => convert_element_type_44
# l__self___blocks_1_norm_in => add_26, mul_23
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
    size_hints=[524288, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: '*bf16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_native_layer_norm_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((24*(x0 % 4)) + (24*(tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))) + (96*((x1 % 196) % 14)) + (1344*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (1344*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*((x1 % 196) // 14)) + (75264*(x1 // 196)) + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + (16*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr2 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r2 + (24*x3)), rmask, other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp12 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tmp11 - tmp21
    tmp29 = 24.0
    tmp30 = tmp27 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp34 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr3 + (r2 + (24*x3)), tmp39, rmask)
    tl.store(out_ptr4 + (r2 + (24*x3)), tmp44, rmask)
''')
