

# Original file: ./lcnet_050___60.0/lcnet_050___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/fc/cfcramjgm6qnjrhzurv7665clqpzidxn2fsk6j4mwtdufz5vrk5g.py
# Source Nodes: [batch_norm_24, getattr_getattr_l__mod___blocks___5_____0___bn2_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# batch_norm_24 => add_74, mul_98, mul_99, sub_24
# getattr_getattr_l__mod___blocks___5_____0___bn2_act => add_75, clamp_max_25, clamp_min_25, convert_element_type_126, div_25, mul_100
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 3.0
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 6.0
    tmp17 = triton_helpers.minimum(tmp15, tmp16)
    tmp18 = tmp11 * tmp17
    tmp19 = tmp18 / tmp16
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp20, None)
''')
