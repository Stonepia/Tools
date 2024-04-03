

# Original file: ./lcnet_050___60.0/lcnet_050___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/a4/ca45wezczcmhfojn35gxxbaoijikcgjvnytqxxzhppolkugjz7i2.py
# Source Nodes: [batch_norm_11, getattr_getattr_l__self___blocks___3_____0___bn1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# batch_norm_11 => add_34, mul_45, mul_46, sub_11
# getattr_getattr_l__self___blocks___3_____0___bn1_act => add_35, clamp_max_11, clamp_min_11, convert_element_type_72, div_11, mul_47
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = 3.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = 6.0
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tmp9 * tmp15
    tmp17 = tmp16 / tmp14
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')
