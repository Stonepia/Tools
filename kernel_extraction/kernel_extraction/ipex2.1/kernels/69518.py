

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ma/cmae4t4fjfqy4rxjpwpf7juod4tlqwu7dsd2ux2yrjny6v2tgnrk.py
# Source Nodes: [group_norm_15], Original ATen: [aten.native_group_norm]
# group_norm_15 => add_53, convert_element_type_75, mul_67
triton_poi_fused_native_group_norm_27 = async_compile.triton('triton_poi_fused_native_group_norm_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16384, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_native_group_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (y1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 150528.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (192*x2) + (150528*y1)), tmp23, xmask)
''')