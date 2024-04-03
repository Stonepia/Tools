

# Original file: ./densenet121___60.0/densenet121___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/id/cidi4tovpndwxate3aoakb4bintbso7cawwn7qdvqxr7cf4bg64z.py
# Source Nodes: [cat_100, cat_101, cat_103, cat_79, cat_80, cat_81, cat_82, cat_83, cat_84, cat_85, cat_86, cat_87, cat_88, cat_89, cat_90, cat_91, cat_92, cat_93, cat_94, cat_95, cat_96, cat_97, cat_98, cat_99, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_100 => cat_20
# cat_101 => cat_19
# cat_103 => clone_2
# cat_79 => cat_41
# cat_80 => cat_40
# cat_81 => cat_39
# cat_82 => cat_38
# cat_83 => cat_37
# cat_84 => cat_36
# cat_85 => cat_35
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
# cat_99 => cat_21
# l__mod___features_denseblock3_denselayer1_norm1 => add_79, mul_118, mul_119, sub_39
# l__mod___features_denseblock3_denselayer1_relu1 => relu_39
# l__mod___features_transition2_pool => avg_pool2d_1
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 14
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (14336*x2)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (14336*x2)), None)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (14336*x2)), None)
    tmp5 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (14336*x2)), None)
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
    tl.store(out_ptr1 + (x0 + (288*x4)), tmp8, None)
    tl.store(out_ptr2 + (x0 + (320*x4)), tmp8, None)
    tl.store(out_ptr3 + (x0 + (352*x4)), tmp8, None)
    tl.store(out_ptr4 + (x0 + (384*x4)), tmp8, None)
    tl.store(out_ptr5 + (x0 + (416*x4)), tmp8, None)
    tl.store(out_ptr6 + (x0 + (448*x4)), tmp8, None)
    tl.store(out_ptr7 + (x0 + (480*x4)), tmp8, None)
    tl.store(out_ptr8 + (x0 + (512*x4)), tmp8, None)
    tl.store(out_ptr9 + (x0 + (544*x4)), tmp8, None)
    tl.store(out_ptr10 + (x0 + (576*x4)), tmp8, None)
    tl.store(out_ptr11 + (x0 + (608*x4)), tmp8, None)
    tl.store(out_ptr12 + (x0 + (640*x4)), tmp8, None)
    tl.store(out_ptr13 + (x0 + (672*x4)), tmp8, None)
    tl.store(out_ptr14 + (x0 + (704*x4)), tmp8, None)
    tl.store(out_ptr15 + (x0 + (736*x4)), tmp8, None)
    tl.store(out_ptr16 + (x0 + (768*x4)), tmp8, None)
    tl.store(out_ptr17 + (x0 + (800*x4)), tmp8, None)
    tl.store(out_ptr18 + (x0 + (832*x4)), tmp8, None)
    tl.store(out_ptr19 + (x0 + (864*x4)), tmp8, None)
    tl.store(out_ptr20 + (x0 + (896*x4)), tmp8, None)
    tl.store(out_ptr21 + (x0 + (928*x4)), tmp8, None)
    tl.store(out_ptr22 + (x0 + (960*x4)), tmp8, None)
    tl.store(out_ptr23 + (x0 + (992*x4)), tmp8, None)
    tl.store(out_ptr24 + (x0 + (1024*x4)), tmp8, None)
''')
