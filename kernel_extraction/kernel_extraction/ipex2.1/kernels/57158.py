

# Original file: ./Background_Matting___60.0/Background_Matting___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/e6/ce6pdx57dw5ytf43ri2sntczjs3xtu7wf5jde36czdq32lzrgaok.py
# Source Nodes: [l__mod___model_enc_seg_0], Original ATen: [aten.reflection_pad2d]
# l__mod___model_enc_seg_0 => reflection_pad2d_2
triton_poi_fused_reflection_pad2d_3 = async_compile.triton('triton_poi_fused_reflection_pad2d_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_reflection_pad2d_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268324
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 518)
    x0 = xindex % 518
    x2 = xindex
    tmp0 = (-3) + x1
    tmp1 = tl.abs(tmp0)
    tmp2 = tl.full([1], 511, tl.int32)
    tmp3 = tmp2 - tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-3) + x0
    tmp7 = tl.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (tmp10 + (512*tmp5)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''')
