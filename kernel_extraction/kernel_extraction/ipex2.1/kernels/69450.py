

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/kr/ckrry6iebmy2qxisaoxcb5jky4vejk2sbxbo37kzranuae5adkjz.py
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
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
    tmp0 = tl.load(in_ptr0 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + ((48*x0) + (192*((r3 + (168*x1)) % 784)) + (150528*x2) + ((r3 + (168*x1)) // 784)), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + ((48*x0) + ((r3 + (168*x1)) // 784)), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tl.full([XBLOCK, 1], 168, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr2 + (x4), tmp15, None)
''')
