

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/qh/cqhbxkju42x3y3bwgey43cpd4yrcyqo4oln7vk2e2iagrynqk3kz.py
# Source Nodes: [group_norm_61], Original ATen: [aten.native_group_norm]
# group_norm_61 => var_mean_61
triton_per_fused_native_group_norm_50 = async_compile.triton('triton_per_fused_native_group_norm_50', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_50', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_group_norm_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18880
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.load(in_ptr1 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp8 + tmp14
    tmp16 = tl.where(tmp6, tmp15, 0)
    tmp17 = tl.where(tmp2, tmp16, 0)
    tmp18 = 0.0
    tmp19 = tl.where(tmp6, tmp18, 0)
    tmp20 = tl.where(tmp2, tmp19, 0)
    tmp21 = 1.0
    tmp22 = tl.where(tmp6, tmp21, 0)
    tmp23 = tl.where(tmp2, tmp22, 0)
    tmp24 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp25 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29 = tl.where(rmask & xmask, tmp25, 0)
    tmp30 = tl.where(rmask & xmask, tmp26, 0)
    tmp31, tmp32, tmp33 = triton_helpers.welford(tmp28, tmp29, tmp30, 1)
    tmp34 = tmp31[:, None]
    tmp35 = tmp32[:, None]
    tmp36 = tmp33[:, None]
    tl.store(out_ptr0 + (x5), tmp34, xmask)
    tl.store(out_ptr1 + (x5), tmp35, xmask)
    tl.store(out_ptr2 + (x5), tmp36, xmask)
''')
