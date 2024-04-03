

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/fd/cfdmqrpoq2exyltyxn5ucuapxvs7dqmmzdmmwjrvr46vhwcsmvug.py
# Source Nodes: [add, add_1, mul, mul_1, sub], Original ATen: [aten.add, aten.mul, aten.sub]
# add => add_2
# add_1 => add_6
# mul => mul_2
# mul_1 => mul_8
# sub => sub_1
triton_poi_fused_add_mul_sub_10 = async_compile.triton('triton_poi_fused_add_mul_sub_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144, 128], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_sub_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 200704
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + (96*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x2 + (96*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (y0 + (3136*x2) + (301056*y1)), tmp10, xmask)
''')
