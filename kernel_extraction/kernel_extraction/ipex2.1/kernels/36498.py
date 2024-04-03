

# Original file: ./resmlp_12_224___60.0/resmlp_12_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/k7/ck7lrebcouiyntv4xuuacdjujxsxel2b3ujwyjxsbwxihny3fie6.py
# Source Nodes: [add, add_1, addcmul_2, getattr_l__mod___blocks___0___mlp_channels_drop2, mul, mul_1], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mul]
# add => add_1
# add_1 => add_5
# addcmul_2 => add_6, convert_element_type_12, convert_element_type_13, mul_10
# getattr_l__mod___blocks___0___mlp_channels_drop2 => clone_2
# mul => mul_2
# mul_1 => mul_8
triton_poi_fused_add_addcmul_clone_mul_5 = async_compile.triton('triton_poi_fused_add_addcmul_clone_mul_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addcmul_clone_mul_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_addcmul_clone_mul_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = tmp8.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp8, xmask & ymask)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp14, xmask & ymask)
''')
