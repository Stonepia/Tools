

# Original file: ./cspdarknet53___60.0/cspdarknet53___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/j2/cj2gckmzt3vklhd4bs7yq4pgbtryy4shdju5btbna7rkbecj23lq.py
# Source Nodes: [add_19, batch_norm_58, getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# add_19 => add_137
# batch_norm_58 => add_136, mul_233, mul_234, sub_58
# getattr_getattr_l__mod___stages___4___blocks___0___conv2_bn_act => convert_element_type_294, gt_58, mul_235, where_58
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_50', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_50', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 64
    y3 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x1 + (512*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr5 + (32768 + y2 + (64*x1) + (65536*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.01
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr1 + (x1 + (512*y0)), tmp19, xmask)
''')
