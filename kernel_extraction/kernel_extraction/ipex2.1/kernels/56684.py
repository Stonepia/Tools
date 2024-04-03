

# Original file: ./YituTechConvBert__0_forward_169.0/YituTechConvBert__0_forward_169.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/6o/c6ool7vxuzn3s4kokzd2ifapqevkhihhavecemx56qwwytb5klix.py
# Source Nodes: [add_2, softmax_1], Original ATen: [aten._softmax, aten.add]
# add_2 => div_1
# softmax_1 => convert_element_type_7, convert_element_type_8, div_2, exp_1, sub_3
triton_poi_fused__softmax_add_10 = async_compile.triton('triton_poi_fused__softmax_add_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128, 262144], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 512)
    y0 = yindex % 6
    y1 = (yindex // 6)
    tmp0 = tl.load(in_ptr0 + (x5 + (262144*y4)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x3 + (512*y4)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x3 + (512*y4)), ymask, eviction_policy='evict_last')
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (6*x5) + (1572864*y1)), tmp9, ymask)
''')
