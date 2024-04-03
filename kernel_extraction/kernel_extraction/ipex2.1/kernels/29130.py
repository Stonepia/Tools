

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/rs/crsqhm6ygdcp2fu7cy2ocokl2gr224jdsg2rcvaaivm7njhcusxg.py
# Source Nodes: [add_14, add_15, add_16, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___mlp_drop2, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_14 => add_102
# add_15 => add_105
# add_16 => add_108
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_proj_drop => clone_56
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___mlp_drop2 => clone_58
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_proj_drop => clone_63
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2 => add_109, add_110, mul_166, mul_167, rsqrt_17, sub_54, var_mean_17
triton_red_fused_add_clone_native_layer_norm_42 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_42', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_42', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    x3 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((240*(((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + ((x1 % 4) % 2)) % 8)) + (1920*((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 8) % 8)) + (15360*((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 15360) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 64) % 240)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (240*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (240*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r2 + (240*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + ((240*(((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + ((x1 % 4) % 2)) % 8)) + (1920*((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 8) % 8)) + (15360*((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 15360) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 4)) + (8*((x1 % 4) // 2)) + (16*((((4*x0) + (64*r2) + (15360*(x1 // 4)) + (x1 % 4)) // 16) % 61440)) + ((x1 % 4) % 2)) // 64) % 240)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r2 + (240*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r2 + (240*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r2 + (240*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 - tmp8
        tmp19 = 240.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-05
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr2 + (r2 + (240*x3)), tmp28, rmask)
''')
