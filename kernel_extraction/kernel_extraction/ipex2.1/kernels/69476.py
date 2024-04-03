

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/rw/crwyeq5fhilbrihywhiinznhyidjz3yi3ry3lamg7ppsk3t7smlr.py
# Source Nodes: [group_norm_60], Original ATen: [aten.native_group_norm]
# group_norm_60 => var_mean_60
triton_per_fused_native_group_norm_46 = async_compile.triton('triton_per_fused_native_group_norm_46', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_group_norm_46(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, other=0.0)
    tmp8 = tl.where(tmp6, tmp7, 0)
    tmp9 = tl.where(tmp2, tmp8, 0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp6, tmp10, 0)
    tmp12 = tl.where(tmp2, tmp11, 0)
    tmp13 = 1.0
    tmp14 = tl.where(tmp6, tmp13, 0)
    tmp15 = tl.where(tmp2, tmp14, 0)
    tmp16 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp17 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp16, 0)
    tmp21 = tl.where(rmask & xmask, tmp17, 0)
    tmp22 = tl.where(rmask & xmask, tmp18, 0)
    tmp23, tmp24, tmp25 = triton_helpers.welford(tmp20, tmp21, tmp22, 1)
    tmp26 = tmp23[:, None]
    tmp27 = tmp24[:, None]
    tmp28 = tmp25[:, None]
    tl.store(out_ptr0 + (x5), tmp26, xmask)
    tl.store(out_ptr1 + (x5), tmp27, xmask)
    tl.store(out_ptr2 + (x5), tmp28, xmask)
''')
