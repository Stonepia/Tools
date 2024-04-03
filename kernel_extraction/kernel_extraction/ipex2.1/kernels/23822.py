

# Original file: ./coat_lite_mini___60.0/coat_lite_mini___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/u2/cu2yyaao4qwg56dqgr6cp2t7j6svqoujhq7mxkuprhsioq2vkmoz.py
# Source Nodes: [reshape_12], Original ATen: [aten.clone]
# reshape_12 => clone_20
triton_poi_fused_clone_35 = async_compile.triton('triton_poi_fused_clone_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_35', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12861440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    x2 = (xindex // 128) % 785
    x3 = (xindex // 100480)
    x4 = xindex % 128
    x5 = (xindex // 128)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (12560*x1) + (100480*x3)), None).to(tl.float32)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = (-1) + x2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x4 + (384*x5)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x4 + (128*(((-1) + x2) % 784)) + (100352*x3)), tmp5, other=0.0).to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.where(tmp5, tmp8, 0.0)
    tmp10 = tmp2 + tmp9
    tl.store(out_ptr0 + (x6), tmp10, None)
''')
