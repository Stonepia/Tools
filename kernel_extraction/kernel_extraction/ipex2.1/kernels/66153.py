

# Original file: ./fbnetv3_b___60.0/fbnetv3_b___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/oz/cozzvj76ibdedgx6vizpjrc3tokfg4lzxooa5ovzppqsjalzstlt.py
# Source Nodes: [batch_norm_32, getattr_getattr_l__mod___blocks___3_____0___bn1_act], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# batch_norm_32 => add_105, mul_128, mul_129, sub_32
# getattr_getattr_l__mod___blocks___3_____0___bn1_act => add_106, clamp_max_31, clamp_min_31, convert_element_type_162, div_31, mul_130
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26214400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 200
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