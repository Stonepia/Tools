

# Original file: ./hf_distil_whisper___60.0/hf_distil_whisper___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/j6/cj6ovpygwzrftntzrlb3t6l5gl36mckmajevj6px3o2h4v7b4j74.py
# Source Nodes: [add, l__mod___encoder_layers_0_self_attn_layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add => add_2
# l__mod___encoder_layers_0_self_attn_layer_norm => add_3, add_4, clone_1, convert_element_type_4, convert_element_type_5, mul_6, mul_7, rsqrt, sub, var_mean
triton_poi_fused_add_native_layer_norm_3 = async_compile.triton('triton_poi_fused_add_native_layer_norm_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1500
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1500*x1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x1 + (1024*y0)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = 1024.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp17 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp1.to(tl.float32)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr0 + (x1 + (1024*y0)), tmp30, xmask & ymask)
''')
