

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ig/cigdqt2skoxpwj2harmto3iapmdkhqslytr3omy4xww24a6npkvy.py
# Source Nodes: [reshape_21], Original ATen: [aten.clone]
# reshape_21 => clone_49
triton_poi_fused_clone_41 = async_compile.triton('triton_poi_fused_clone_41', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4, 1048576], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 786432
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + ((192*(((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)) + (12288*y0) + (24576*y1) + (49152*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((64*y0) + (128*y1) + (256*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + ((64*y0) + (128*y1) + (256*((y0 + (2*y1) + (4*x2) + (32*x3)) // 49152)) + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 4) % 64)), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (((y0 + (2*y1) + (4*x2) + (32*x3)) // 256) % 192), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 192.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (2*x2) + (16*y1) + (32*x3)), tmp17, ymask)
''')
