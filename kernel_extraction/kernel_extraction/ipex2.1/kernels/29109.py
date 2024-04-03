

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/pc/cpcooosqeo64gg4fae7sajqlt4b36k44i6axrnmnixx6iaje3mqp.py
# Source Nodes: [add_6, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_6 => add_64
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_proj_drop => clone_24
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => add_65, add_66, mul_106, mul_107, rsqrt_6, sub_32, var_mean_6
triton_red_fused_add_clone_native_layer_norm_21 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_21', '''
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
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((192*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) % 16)) + (3072*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + ((192*(((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + ((x1 % 4) % 2)) % 16)) + (3072*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x0) + (x1 % 4)) // 4) % 8)) + (16*((x1 % 4) // 2)) + (32*((((4*x0) + (256*r2) + (49152*(x1 // 4)) + (x1 % 4)) // 32) % 98304)) + ((x1 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2 + (192*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 192.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tl.store(out_ptr2 + (r2 + (192*x3)), tmp20, rmask)
''')
