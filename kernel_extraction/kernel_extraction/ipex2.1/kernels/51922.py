

# Original file: ./pytorch_CycleGAN_and_pix2pix___60.0/pytorch_CycleGAN_and_pix2pix___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/wi/cwiasxix5uqh7z272gkvtlwnlmljtxl4zi2yjmbzdefx7axzebtk.py
# Source Nodes: [add_7, add_8, getattr_l__self___model___17___conv_block_6, getattr_l__self___model___18___conv_block_6, l__self___model_19, l__self___model_20, l__self___model_21, l__self___model_22], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.relu]
# add_7 => add_26
# add_8 => add_29
# getattr_l__self___model___17___conv_block_6 => add_25, convert_element_type_75, convert_element_type_76, mul_18, rsqrt_18, sub_18, var_mean_18
# getattr_l__self___model___18___conv_block_6 => add_28, convert_element_type_83, convert_element_type_84, mul_20, rsqrt_20, sub_20, var_mean_20
# l__self___model_19 => convolution_21
# l__self___model_20 => add_30, convert_element_type_87, convert_element_type_88, mul_21, rsqrt_21, sub_21, var_mean_21
# l__self___model_21 => relu_12
# l__self___model_22 => convolution_22
triton_poi_fused__native_batch_norm_legit_add_convolution_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_add_convolution_relu_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192, 16], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_add_convolution_relu_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_add_convolution_relu_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')
