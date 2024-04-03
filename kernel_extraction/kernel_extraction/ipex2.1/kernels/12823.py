

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/rc/crcbmk7lly6lnhbclyrjm47wzimr4au7pr4mnyx7aqeh3h5ztnub.py
# Source Nodes: [add_3, argmax, mod, mul_1], Original ATen: [aten.add, aten.argmax, aten.mul, aten.remainder]
# add_3 => add_10
# argmax => argmax
# mod => remainder
# mul_1 => mul_7
triton_per_fused_add_argmax_mul_remainder_18 = async_compile.triton('triton_per_fused_add_argmax_mul_remainder_18', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_argmax_mul_remainder_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_argmax_mul_remainder_18(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 4096
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = tl.broadcast_to(rindex, tmp3.shape)
    _, tmp2_tmp = triton_helpers.max_with_index(tmp3, tmp4, 1)
    tmp2 = tmp2_tmp[:, None]
    tmp5 = tl.full([1, 1], 0, tl.int64)
    tmp6 = tmp2 + tmp5
    tmp7 = tl.full([1, 1], 4096, tl.int64)
    tmp8 = tmp6 * tmp7
    tmp9 = x2
    tmp10 = tmp8 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''')
