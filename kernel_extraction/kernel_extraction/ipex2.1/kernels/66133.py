

# Original file: ./fbnetv3_b___60.0/fbnetv3_b___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/6q/c6qovukolaxeenwo2nbso6v5utfivmdm4vrdwc3ebrotzldloh3h.py
# Source Nodes: [batch_norm_83, getattr_getattr_l__self___blocks___5_____6___bn1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# batch_norm_83 => add_279, mul_339, mul_340, sub_83
# getattr_getattr_l__self___blocks___5_____6___bn1_act => add_280, clamp_max_89, clamp_min_89, convert_element_type_584, div_89, mul_341
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_31', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9043968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1104
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
