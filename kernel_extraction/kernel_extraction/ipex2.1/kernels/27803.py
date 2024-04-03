

# Original file: ./detectron2_fcos_r_50_fpn__79_inference_119.59/detectron2_fcos_r_50_fpn__79_inference_119.59_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/im/cim5bfi6xfc4lbbvboymwdhyfwhzmaad3qsau7fzqzl74diryf3v.py
# Source Nodes: [max_1], Original ATen: [aten.max]
# max_1 => max_1
triton_red_fused_max_0 = async_compile.triton('triton_red_fused_max_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_max_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_max_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3
    rnumel = 6667
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = r1 + (6667*x0)
        tmp1 = tl.full([1, 1], 20000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((r1 + (6667*x0)) % 20000), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.where(tmp2, tmp3, float("-inf"))
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')
