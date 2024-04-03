

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/26/c26gbdogwmradkehwwwiojoxtrlkiwpy3jjlnz4qwjpa4zm7hqur.py
# Source Nodes: [l__mod___blocks_0_norm_in, reshape_2], Original ATen: [aten.clone, aten.native_layer_norm]
# l__mod___blocks_0_norm_in => add_8, add_9, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
# reshape_2 => clone_2
triton_per_fused_clone_native_layer_norm_3 = async_compile.triton('triton_per_fused_clone_native_layer_norm_3', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + ((24*(x0 % 4)) + (24*(tl.where(((4*((x1 % 196) % 14)) + (x0 % 4)) >= 0, 0, 56))) + (96*((x1 % 196) % 14)) + (1344*((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) + (1344*(tl.where(((4*((x1 % 196) // 14)) + ((((4*(x0 // 4)) + (x0 % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*((x1 % 196) // 14)) + (75264*(x1 // 196)) + ((((4*(x0 // 4)) + (16*r2) + (x0 % 4)) // 16) % 24)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*r2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 24, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 24.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(out_ptr2 + (r2 + (24*x3)), tmp29, rmask)
''')
