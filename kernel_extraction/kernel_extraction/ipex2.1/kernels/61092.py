

# Original file: ./timm_efficientnet___60.0/timm_efficientnet___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/tv/ctvfg3og5d4hubp4nry5i35e2f43kjrl7g7dsa3wmjxodjlt3f2a.py
# Source Nodes: [batch_norm_1, getattr_getattr_l__self___blocks___0_____0___bn1_act, mean], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# batch_norm_1 => add_3, mul_5, mul_6, sub_1
# getattr_getattr_l__self___blocks___0_____0___bn1_act => convert_element_type_12, mul_7, sigmoid_1
# mean => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_silu_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_mean_silu_3', '''
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
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_silu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_mean_silu_3(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 28
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (896*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 12544.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp7, None)
''')
