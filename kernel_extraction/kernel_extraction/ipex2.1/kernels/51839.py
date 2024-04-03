

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/5a/c5a6dszgw6npdgowddfv5vkv4uxwydy6b4cqt57ihkggl74xmxq3.py
# Source Nodes: [add_7, add_8, getattr_l__mod___model___17___conv_block_6, getattr_l__mod___model___18___conv_block_6, l__mod___model_19], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution]
# add_7 => add_26
# add_8 => add_29
# getattr_l__mod___model___17___conv_block_6 => add_25, convert_element_type_36, convert_element_type_37, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__mod___model___18___conv_block_6 => add_28, convert_element_type_40, convert_element_type_41, mul_20, rsqrt_20, sub_20, var_mean_20
# l__mod___model_19 => convolution_21
triton_poi_fused__native_batch_norm_legit_add_convolution_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_convolution_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 16], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_convolution_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')
