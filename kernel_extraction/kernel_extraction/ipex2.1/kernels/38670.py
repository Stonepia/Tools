

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/y7/cy7qlqrqiynk73yj6ewfcle4mx5qq24lrin6vjlqpcvms4rjvfda.py
# Source Nodes: [abs_3, mean_2, sub_19], Original ATen: [aten.abs, aten.mean, aten.sub]
# abs_3 => abs_8
# mean_2 => mean_8
# sub_19 => sub_137
triton_per_fused_abs_mean_sub_61 = async_compile.triton('triton_per_fused_abs_mean_sub_61', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_sub_61', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_mean_sub_61(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 11584
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = r2 + (128*x3)
    tmp1 = tl.full([1, 1], 1482624, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + ((4*((r2 + (128*x0) + (8192*x1)) % 351)) + (1408*(((r2 + (128*x0) + (8192*x1)) // 351) % 352)) + (495616*(((r2 + (128*x0) + (8192*x1)) // 247104) % 6)) + (((r2 + (128*x0) + (8192*x1)) // 123552) % 2)), rmask & tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.1
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (4 + (4*((r2 + (128*x0) + (8192*x1)) % 351)) + (1408*(((r2 + (128*x0) + (8192*x1)) // 351) % 352)) + (495616*(((r2 + (128*x0) + (8192*x1)) // 247104) % 6)) + (((r2 + (128*x0) + (8192*x1)) // 123552) % 2)), rmask & tmp2 & xmask, other=0.0).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 > tmp5
    tmp14 = tmp12 * tmp7
    tmp15 = tl.where(tmp13, tmp12, tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp10 - tmp16
    tmp18 = tl.abs(tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.where(tmp2, tmp19, 0)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''')
