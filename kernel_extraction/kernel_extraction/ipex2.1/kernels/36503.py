

# Original file: ./resmlp_12_224___60.0/resmlp_12_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/ud/cudk63jeq5yf7b3k3ibjqd4tcxbxjhbm235vtic3b2yr4qcqdfb7.py
# Source Nodes: [add, addcmul_1, getattr_l__self___blocks___0___mlp_channels_fc1, mul], Original ATen: [aten._to_copy, aten.add, aten.addcmul, aten.mul]
# add => add_1
# addcmul_1 => add_2, mul_4
# getattr_l__self___blocks___0___mlp_channels_fc1 => convert_element_type_7
# mul => mul_2
triton_poi_fused__to_copy_add_addcmul_mul_2 = async_compile.triton('triton_poi_fused__to_copy_add_addcmul_mul_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addcmul_mul_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_addcmul_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp9 = tmp1 * tmp8
    tmp10 = tmp0 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp11, xmask & ymask)
''')
