

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/vd/cvdhulfjktlr2crazhysrzopcr53t5x6fxjcn7ayyzdnyajwlgpe.py
# Source Nodes: [reshape_32], Original ATen: [aten.clone]
# reshape_32 => clone_74
triton_poi_fused_clone_46 = async_compile.triton('triton_poi_fused_clone_46', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4, 262144], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_46', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 245760
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 4
    x3 = (xindex // 4)
    y0 = yindex % 2
    y1 = (yindex // 2)
    tmp0 = tl.load(in_ptr0 + ((240*(((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)) + (3840*y0) + (7680*y1) + (15360*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((240*(((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)) + (3840*y0) + (7680*y1) + (15360*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((240*(((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)) + (3840*y0) + (7680*y1) + (15360*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + ((16*y0) + (32*y1) + (64*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + ((16*y0) + (32*y1) + (64*((y0 + (2*y1) + (4*x2) + (16*x3)) // 15360)) + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 4) % 16)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (((y0 + (2*y1) + (4*x2) + (16*x3)) // 64) % 240), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 240.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (y0 + (2*x2) + (8*y1) + (16*x3)), tmp17, ymask)
''')
