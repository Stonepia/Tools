

# Original file: ./pytorch_stargan___60.0/pytorch_stargan___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/lo/clolte4qh4fgavgjyun4tle36cr4hvsix7topt5xchzgdihtlrvo.py
# Source Nodes: [add_5, l__mod___main_15], Original ATen: [aten.add, aten.convolution]
# add_5 => add_35
# l__mod___main_15 => convolution_15
triton_poi_fused_add_convolution_11 = async_compile.triton('triton_poi_fused_add_convolution_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_convolution_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 1024)
    y0 = yindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (x2 + (256*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (x2 + (256*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 - tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sqrt(tmp9)
    tmp11 = 1 / tmp10
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp5 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp0 + tmp21
    tl.store(out_ptr0 + (y0 + (1024*x2) + (262144*y1)), tmp22, xmask)
''')
