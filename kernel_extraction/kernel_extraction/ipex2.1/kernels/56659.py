

# Original file: ./YituTechConvBert__0_forward_169.0/YituTechConvBert__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/r2/cr223fksiti5hpwm373go63bqk5u3vf6tfknnfvqugoq3prpxr56.py
# Source Nodes: [reshape_3], Original ATen: [aten.clone]
# reshape_3 => clone_2
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304, 16], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3145728
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = (yindex // 384) % 512
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp2 = tl.where(tmp1 < 0, tmp1 + 1, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 1), "index out of bounds: 0 <= tmp2 < 1")
    tmp3 = (-4) + x3 + y1
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1, 1], 512, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.load(in_ptr1 + ((-1536) + y4 + (384*x3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.where(tmp8, tmp10, 0.0)
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x3 + (9*y4)), tmp12, xmask)
''')
