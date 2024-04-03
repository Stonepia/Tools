

# Original file: ./DALLE2_pytorch__31_inference_71.11/DALLE2_pytorch__31_inference_71.11.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/5e/c5evh4wf32rivzpidgdaj7ola54mfvnnofzvd5iqruhxtkbnd3pb.py
# Source Nodes: [mul_12, normalize_1], Original ATen: [aten.div, aten.mul, aten.norm]
# mul_12 => mul_12
# normalize_1 => div_1, pow_3, sum_2
triton_per_fused_div_mul_norm_12 = async_compile.triton('triton_per_fused_div_mul_norm_12', '''
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
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_norm_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_div_mul_norm_12(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 522
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1e-12
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp0 / tmp8
    tmp10 = 4.0
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr1 + (r1 + (64*x0)), tmp11, rmask & xmask)
''')
