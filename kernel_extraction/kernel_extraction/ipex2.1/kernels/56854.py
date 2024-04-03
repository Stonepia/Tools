

# Original file: ./YituTechConvBert__0_backward_207.1/YituTechConvBert__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/qi/cqieyc7fzjmmyk4pqk2l62r53prjk5zevzg4phdaxw62ll2l3kuz.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_34 = async_compile.triton('triton_poi_fused_add_mul_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 512
    y3 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x1 + (384*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + (384*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (y2 + (512*x1) + (196608*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + ((64*y2) + (32768*(x1 // 64)) + (196608*y3) + (x1 % 64)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp0 * tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x1 + (384*y0)), tmp2, xmask)
    tl.store(out_ptr1 + (x1 + (384*y0)), tmp10, xmask)
''')
