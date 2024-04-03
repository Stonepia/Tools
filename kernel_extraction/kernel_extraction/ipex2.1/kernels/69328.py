

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/tf/ctfrygz25ljv67ccjekz6h2lf5pddvktmpmvrdf7zwhvwnsuutvd.py
# Source Nodes: [group_norm_13], Original ATen: [aten.native_group_norm]
# group_norm_13 => var_mean_13
triton_per_fused_native_group_norm_20 = async_compile.triton('triton_per_fused_native_group_norm_20', '''
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
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_native_group_norm_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 168
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 4
    x1 = (xindex // 4) % 224
    x2 = (xindex // 896)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + ((48*x0) + ((r3 + (168*x1)) // 784)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 168, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp18, None)
    tl.store(out_ptr1 + (x4), tmp24, None)
    tl.store(out_ptr2 + (x4), tmp17, None)
''')
