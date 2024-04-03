

# Original file: ./detectron2_maskrcnn__66_inference_106.46/detectron2_maskrcnn__66_inference_106.46_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/gz/cgzjbkemu36ul6yzc4zpdn2qqnnjzgcnxfkppvtgpkylbtt2quhr.py
# Source Nodes: [all_1, all_2, and_, isfinite_1], Original ATen: [aten.abs, aten.all, aten.bitwise_and, aten.eq, aten.mul, aten.ne]
# all_1 => logical_not_1
# all_2 => any_2, logical_not_2, logical_not_3
# and_ => bitwise_and
# isfinite_1 => abs_2, eq_1, mul_1, ne_1
triton_per_fused_abs_all_bitwise_and_eq_mul_ne_1 = async_compile.triton('triton_per_fused_abs_all_bitwise_and_eq_mul_ne_1', '''
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_all_bitwise_and_eq_mul_ne_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_all_bitwise_and_eq_mul_ne_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 81
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (81*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == tmp0
    tmp2 = tl.abs(tmp0)
    tmp3 = float("inf")
    tmp4 = tmp2 != tmp3
    tmp5 = tmp1 & tmp4
    tmp6 = tmp5 == 0
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.any(tmp11, 1)[:, None]
    tmp14 = tmp13 == 0
    tmp15 = tmp12 == 0
    tmp16 = tmp14 & tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)
''')
