

# Original file: ./cspdarknet53___60.0/cspdarknet53___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/zz/czzc5cjjbp22cqemevffmb7zrtvnut7ddjziiumaua3ectd4dznj.py
# Source Nodes: [add, batch_norm_4, getattr_getattr_l__self___stages___0___blocks___0___conv2_bn_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# add => add_10
# batch_norm_4 => add_9, mul_17, mul_18, sub_4
# getattr_getattr_l__self___stages___0___blocks___0___conv2_bn_act => convert_element_type_30, gt_4, mul_19, where_4
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576, 64], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = yindex % 16384
    y3 = (yindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x1 + (64*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (1048576 + y2 + (16384*x1) + (2097152*y3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = 0.01
    tmp13 = tmp9 * tmp12
    tmp14 = tl.where(tmp11, tmp9, tmp13)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr1 + (x1 + (64*y0)), tmp17, xmask)
''')
