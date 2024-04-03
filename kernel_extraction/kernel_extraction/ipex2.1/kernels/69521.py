

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/f7/cf7r3rzbmx4zj545lobkwyeozdr4ko3bsgp36cjlw6jhkyjbyf67.py
# Source Nodes: [group_norm_24], Original ATen: [aten.native_group_norm]
# group_norm_24 => convert_element_type_120, var_mean_24
triton_per_fused_native_group_norm_30 = async_compile.triton('triton_per_fused_native_group_norm_30', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_group_norm_30(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 46400
    rnumel = 104
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 145
    x1 = (xindex // 145) % 5
    x2 = (xindex // 725)
    x5 = xindex
    tmp0 = r3 + (104*x0)
    tmp1 = tl.full([1, 1], 15053, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (104*x0) + (15053*x1)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (104*x0) + (15053*x1)) % 196)) + (75264*x2) + (((r3 + (104*x0) + (15053*x1)) // 196) % 384)), rmask & tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.where(tmp6, tmp8, 0)
    tmp10 = tl.where(tmp2, tmp9, 0)
    tmp11 = 0.0
    tmp12 = tl.where(tmp6, tmp11, 0)
    tmp13 = tl.where(tmp2, tmp12, 0)
    tmp14 = 1.0
    tmp15 = tl.where(tmp6, tmp14, 0)
    tmp16 = tl.where(tmp2, tmp15, 0)
    tmp17 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp18 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp17, 0)
    tmp22 = tl.where(rmask & xmask, tmp18, 0)
    tmp23 = tl.where(rmask & xmask, tmp19, 0)
    tmp24, tmp25, tmp26 = triton_helpers.welford(tmp21, tmp22, tmp23, 1)
    tmp27 = tmp24[:, None]
    tmp28 = tmp25[:, None]
    tmp29 = tmp26[:, None]
    tl.store(out_ptr0 + (x5), tmp27, xmask)
    tl.store(out_ptr1 + (x5), tmp28, xmask)
    tl.store(out_ptr2 + (x5), tmp29, xmask)
''')
