

# Original file: ./swin_base_patch4_window7_224___60.0/swin_base_patch4_window7_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/w5/cw5znu4dkb77fbo3j7dsfkbwoxkuuyfh2qtn2reuwz2xdlfy42vk.py
# Source Nodes: [contiguous_12], Original ATen: [aten.clone]
# contiguous_12 => clone_35
triton_poi_fused_clone_24 = async_compile.triton('triton_poi_fused_clone_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x5 = (xindex // 200704)
    x6 = (xindex // 256) % 28
    x7 = (xindex // 7168) % 28
    x1 = (xindex // 256) % 7
    x2 = (xindex // 1792) % 4
    x3 = (xindex // 7168) % 7
    x4 = (xindex // 50176) % 4
    x8 = xindex % 1792
    x9 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((3 + x6) % 28)) + (7168*((((28*((3 + x7) % 28)) + ((3 + x6) % 28)) // 28) % 28)) + (200704*x5)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(((3 + x1 + (7*x2)) % 28) % 7)) + (1792*(((((28*((3 + x3 + (7*x4)) % 28)) + ((3 + x1 + (7*x2)) % 28)) // 28) % 28) % 7)) + (12544*(((3 + x1 + (7*x2)) % 28) // 7)) + (50176*(((((28*((3 + x3 + (7*x4)) % 28)) + ((3 + x1 + (7*x2)) % 28)) // 28) % 28) // 7)) + (200704*x5)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + (256*((3 + x6) % 28)) + (7168*((3 + x7) % 28)) + (200704*x5)), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + ((28*((3 + x7) % 28)) + (784*x5) + ((3 + x6) % 28)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp9 = 256.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp7 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x8 + (1792*x3) + (12544*x2) + (50176*x9)), tmp21, None)
''')
