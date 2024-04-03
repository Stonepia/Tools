

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/df/cdf7bn7ddrfuepsnikkwlqh2foajaeqgjyzltjutqjpakzslq7f7.py
# Source Nodes: [group_norm_63], Original ATen: [aten.native_group_norm]
# group_norm_63 => var_mean_63
triton_red_fused_native_group_norm_57 = async_compile.triton('triton_red_fused_native_group_norm_57', '''
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_57', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_group_norm_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.where(tmp2, tmp9, 0)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, 0)
        tmp13 = 1.0
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp16 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp17 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_combine(
            tmp18_mean, tmp18_m2, tmp18_weight,
            tmp15, tmp16, tmp17
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tl.store(out_ptr1 + (x3), tmp19, xmask)
    tl.store(out_ptr2 + (x3), tmp20, xmask)
''')