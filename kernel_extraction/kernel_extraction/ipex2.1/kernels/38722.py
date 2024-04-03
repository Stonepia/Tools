

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/7w/c7wonilvkqavi3reoa3mtyo4m67wlf2s62u2gxm3sihlbdatdw3a.py
# Source Nodes: [abs_3, mean_2, sub_19], Original ATen: [aten.abs, aten.mean, aten.sub]
# abs_3 => abs_8
# mean_2 => mean_8
# sub_19 => sub_137
triton_per_fused_abs_mean_sub_49 = async_compile.triton('triton_per_fused_abs_mean_sub_49', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_sub_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_mean_sub_49(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr0 + ((4*((r2 + (128*x0) + (8192*x1)) % 351)) + (1408*(((r2 + (128*x0) + (8192*x1)) // 351) % 352)) + (495616*(((r2 + (128*x0) + (8192*x1)) // 247104) % 6)) + (((r2 + (128*x0) + (8192*x1)) // 123552) % 2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr0 + (4 + (4*((r2 + (128*x0) + (8192*x1)) % 351)) + (1408*(((r2 + (128*x0) + (8192*x1)) // 351) % 352)) + (495616*(((r2 + (128*x0) + (8192*x1)) // 247104) % 6)) + (((r2 + (128*x0) + (8192*x1)) // 123552) % 2)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.abs(tmp5)
    tmp7 = tl.where(tmp2, tmp6, 0)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')
