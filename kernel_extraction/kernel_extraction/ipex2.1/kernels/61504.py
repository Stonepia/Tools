

# Original file: ./densenet121___60.0/densenet121___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/t3/ct33nt3ghdlh4xpdvrdesre6a2gledtvcoj72z43xmgfv7sg7e5a.py
# Source Nodes: [cat_117, cat_118, cat_119, cat_120, cat_121, cat_123, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
# cat_120 => cat_2
# cat_121 => cat_1
# cat_123 => clone
# l__mod___features_denseblock1_denselayer1_norm1 => add_3, convert_element_type_5, mul_4, mul_5, sub_1
# l__mod___features_denseblock1_denselayer1_relu1 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (301056*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = triton_helpers.maximum(0, tmp12)
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp13, xmask)
    tl.store(out_ptr1 + (y0 + (128*x2) + (401408*y1)), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + (160*x2) + (501760*y1)), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + (192*x2) + (602112*y1)), tmp0, xmask)
    tl.store(out_ptr4 + (y0 + (224*x2) + (702464*y1)), tmp0, xmask)
    tl.store(out_ptr5 + (y0 + (256*x2) + (802816*y1)), tmp0, xmask)
''')