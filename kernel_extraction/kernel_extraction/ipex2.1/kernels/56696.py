

# Original file: ./YituTechConvBert__0_forward_169.0/YituTechConvBert__0_forward_169.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/mm/cmmumgo76ucyztmbw25asib4s4mimrvym63lvtujmitiiqbknaxj.py
# Source Nodes: [l__mod___convbert_encoder_layer_0_attention_self_conv_kernel_layer, multiply], Original ATen: [aten._unsafe_view, aten.clone, aten.mul]
# l__mod___convbert_encoder_layer_0_attention_self_conv_kernel_layer => clone, view_9
# multiply => mul_5
triton_poi_fused__unsafe_view_clone_mul_2 = async_compile.triton('triton_poi_fused__unsafe_view_clone_mul_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_mul_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__unsafe_view_clone_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + ((512*x1) + (196608*(y0 // 512)) + (y0 % 512)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1 + (384*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x1 + (384*y0)), tmp4, xmask)
''')