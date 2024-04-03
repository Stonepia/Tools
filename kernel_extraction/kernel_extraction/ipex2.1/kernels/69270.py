

# Original file: ./poolformer_m36___60.0/poolformer_m36___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/fd/cfdzhp2favwfeivs7ye7dfk7lkcttxfklmseq2xkuk4bh6l6w6xw.py
# Source Nodes: [add_12, add_13, mul_12, mul_13, sub_6], Original ATen: [aten.add, aten.mul, aten.sub]
# add_12 => add_44
# add_13 => add_48
# mul_12 => mul_56
# mul_13 => mul_62
# sub_6 => sub_19
triton_poi_fused_add_mul_sub_23 = async_compile.triton('triton_poi_fused_add_mul_sub_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_sub_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp8 + tmp12
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp13, xmask)
''')
