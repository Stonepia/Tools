

# Original file: ./detectron2_fasterrcnn_r_101_c4__44_inference_84.24/detectron2_fasterrcnn_r_101_c4__44_inference_84.24.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/fm/cfmvhjhyehkfaoc6jmn6ncxp7jbltml4keifzrtkymwv4bkkoieg.py
# Source Nodes: [all_1, isfinite], Original ATen: [aten.abs, aten.all, aten.eq, aten.mul, aten.ne]
# all_1 => any_1, logical_not
# isfinite => abs_1, eq, mul, ne
triton_per_fused_abs_all_eq_mul_ne_0 = async_compile.triton('triton_per_fused_abs_all_eq_mul_ne_0', '''
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
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i1', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_all_eq_mul_ne_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_all_eq_mul_ne_0(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1000
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tmp0 == tmp0
    tmp2 = tl.abs(tmp0)
    tmp3 = float("inf")
    tmp4 = tmp2 != tmp3
    tmp5 = tmp1 & tmp4
    tmp6 = tmp5 == 0
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')
