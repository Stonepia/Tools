

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/c6/cc6ofa6xgymguy7qn376lcmhni5qf7ki2imel4slx5w7yoyvnmcd.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => add_62, add_63, mul_102, mul_103, rsqrt_5, sub_30, var_mean_5
triton_red_fused_native_layer_norm_15 = async_compile.triton('triton_red_fused_native_layer_norm_15', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_15(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((192*(((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + ((x0 % 4) % 2)) % 16)) + (3072*((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((192*(((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + ((x0 % 4) % 2)) % 16)) + (3072*((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 16) % 16)) + (49152*((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 49152) % 64)) + ((((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 98304)) + ((x0 % 4) % 2)) // 256) % 192)), rmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 192.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + (192*x1) + (12288*x0)), tmp16, rmask)
''')
