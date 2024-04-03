

# Original file: ./pytorch_stargan___60.0/pytorch_stargan___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/it/citihfq73np3al3bxkhhbit7hpm63vj7wzhm5mcgtb7wxmcr2kld.py
# Source Nodes: [getattr_l__mod___main___9___main_0, l__mod___main_8], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___main___9___main_0 => _convolution_pointwise_default_12
# l__mod___main_8 => relu_2
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_poi_fused_convolution_relu_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_convolution_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (262144*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (y3), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sqrt(tmp8)
    tmp10 = 1 / tmp9
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = triton_helpers.maximum(0, tmp20)
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp21, xmask)
    tl.store(out_ptr1 + (y0 + (256*x2) + (262144*y1)), tmp21, xmask)
''')
