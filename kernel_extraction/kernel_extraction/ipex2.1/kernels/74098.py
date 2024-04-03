

# Original file: ./dpn107___60.0/dpn107___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/zl/czlnhtq6luodipyne6dxzhi3poepn7zxfodwvywrp7ibvc2h72we.py
# Source Nodes: [batch_norm_100, batch_norm_101, l__self___features_conv5_1_c1x1_a_bn_act, l__self___features_conv5_1_c1x1_w_s2_bn_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# batch_norm_100 => add_233, convert_element_type_403, mul_301, mul_302, sub_100
# batch_norm_101 => add_235, convert_element_type_407, mul_304, mul_305, sub_101
# l__self___features_conv5_1_c1x1_a_bn_act => relu_101
# l__self___features_conv5_1_c1x1_w_s2_bn_act => relu_100
triton_poi_fused__native_batch_norm_legit_no_training_relu_133 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_133', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_133', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_133(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 77824
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2432
    y1 = (yindex // 2432)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = triton_helpers.maximum(0, tmp10)
    tmp13 = tmp5 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (y0 + (2432*x2) + (476672*y1)), tmp11, xmask)
    tl.store(out_ptr1 + (y0 + (2432*x2) + (476672*y1)), tmp17, xmask)
''')
