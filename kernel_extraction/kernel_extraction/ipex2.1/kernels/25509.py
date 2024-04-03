

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/cd/ccd5sxo4ewqvamhgyckuun6ppdc556pmz2es25varqfyvmr3sash.py
# Source Nodes: [add_197], Original ATen: [aten.add]
# add_197 => add_357
triton_poi_fused_add_34 = async_compile.triton('triton_poi_fused_add_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x1)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = 256.0
    tmp4 = tmp2 / tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp1 - tmp5
    tmp8 = tmp7 / tmp3
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = tl.sqrt(tmp11)
    tmp13 = tmp6 / tmp12
    tmp14 = tmp0 * tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x1 + (4096*y0)), tmp18, ymask)
''')
