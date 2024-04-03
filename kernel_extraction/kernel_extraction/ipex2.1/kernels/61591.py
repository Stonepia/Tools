

# Original file: ./densenet121___60.0/densenet121___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/u6/cu6gacznjaapta7ecnmjh3dw3x5uw5aaj3msx455oiexgbm6snx6.py
# Source Nodes: [cat_62, cat_63, cat_64, cat_65, cat_66, cat_67, cat_68, cat_69, cat_70, cat_71, cat_72, cat_73, cat_74, cat_75, cat_76, cat_78, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
# cat_74 => cat_45
# cat_75 => cat_44
# cat_76 => cat_43
# cat_78 => clone_3
# l__mod___features_denseblock4_denselayer1_norm1 => add_177, convert_element_type_266, mul_265, mul_266, sub_88
# l__mod___features_denseblock4_denselayer1_relu1 => relu_88
# l__mod___features_transition3_pool => avg_pool2d_2
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_89 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_89', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_89', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_89(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 7
    x2 = (xindex // 3584)
    x3 = xindex
    x4 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x1) + (14336*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (14336*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (7168 + x0 + (1024*x1) + (14336*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + (1024*x1) + (14336*x2)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = triton_helpers.maximum(0, tmp20)
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x0 + (544*x4)), tmp8, None)
    tl.store(out_ptr2 + (x0 + (576*x4)), tmp8, None)
    tl.store(out_ptr3 + (x0 + (608*x4)), tmp8, None)
    tl.store(out_ptr4 + (x0 + (640*x4)), tmp8, None)
    tl.store(out_ptr5 + (x0 + (672*x4)), tmp8, None)
    tl.store(out_ptr6 + (x0 + (704*x4)), tmp8, None)
    tl.store(out_ptr7 + (x0 + (736*x4)), tmp8, None)
    tl.store(out_ptr8 + (x0 + (768*x4)), tmp8, None)
    tl.store(out_ptr9 + (x0 + (800*x4)), tmp8, None)
    tl.store(out_ptr10 + (x0 + (832*x4)), tmp8, None)
    tl.store(out_ptr11 + (x0 + (864*x4)), tmp8, None)
    tl.store(out_ptr12 + (x0 + (896*x4)), tmp8, None)
    tl.store(out_ptr13 + (x0 + (928*x4)), tmp8, None)
    tl.store(out_ptr14 + (x0 + (960*x4)), tmp8, None)
    tl.store(out_ptr15 + (x0 + (992*x4)), tmp8, None)
    tl.store(out_ptr16 + (x0 + (1024*x4)), tmp8, None)
''')
