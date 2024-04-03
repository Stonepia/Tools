

# Original file: ./densenet121___60.0/densenet121___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ns/cnst2bfheffu6m4aswj4sgecj22fxkwxov6u5cec6rcsyeebxg6b.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108, cat_109, cat_110, cat_111, cat_112, cat_113, cat_114, cat_116, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# cat_112 => cat_9
# cat_113 => cat_8
# cat_114 => cat_7
# cat_116 => clone_1
# l__mod___features_denseblock2_denselayer1_norm1 => add_29, mul_43, mul_44, sub_14
# l__mod___features_denseblock2_denselayer1_relu1 => relu_14
# l__mod___features_transition1_pool => avg_pool2d
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 28
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14336*x2)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x0 + (160*x4)), tmp8, None)
    tl.store(out_ptr2 + (x0 + (192*x4)), tmp8, None)
    tl.store(out_ptr3 + (x0 + (224*x4)), tmp8, None)
    tl.store(out_ptr4 + (x0 + (256*x4)), tmp8, None)
    tl.store(out_ptr5 + (x0 + (288*x4)), tmp8, None)
    tl.store(out_ptr6 + (x0 + (320*x4)), tmp8, None)
    tl.store(out_ptr7 + (x0 + (352*x4)), tmp8, None)
    tl.store(out_ptr8 + (x0 + (384*x4)), tmp8, None)
    tl.store(out_ptr9 + (x0 + (416*x4)), tmp8, None)
    tl.store(out_ptr10 + (x0 + (448*x4)), tmp8, None)
    tl.store(out_ptr11 + (x0 + (480*x4)), tmp8, None)
    tl.store(out_ptr12 + (x0 + (512*x4)), tmp8, None)
''')
