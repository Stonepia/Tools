

# Original file: ./fbnetv3_b___60.0/fbnetv3_b___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/it/citsbboa2ccz46fbouknmushiybb2i2ulsho5bkw5wvrehgdalhr.py
# Source Nodes: [getattr_getattr_l__self___blocks___5_____6___bn2_act, mean_17], Original ATen: [aten.hardswish, aten.mean]
# getattr_getattr_l__self___blocks___5_____6___bn2_act => add_283, clamp_max_90, clamp_min_90, convert_element_type_590, div_90, mul_345
# mean_17 => mean_17
triton_per_fused_hardswish_mean_33 = async_compile.triton('triton_per_fused_hardswish_mean_33', '''
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
    size_hints=[262144, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_hardswish_mean_33(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 141312
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1104
    x1 = (xindex // 1104)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1104*r2) + (70656*x1)), rmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = 64.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')
