

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/t2/ct2ohvigisdsdqvqmnlxcw7y23q7w3edtr4rdfjb7ymcsv557awj.py
# Source Nodes: [l__self___blocks_0_local_mp_act, l__self___blocks_0_local_mp_bn], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# l__self___blocks_0_local_mp_act => add_20, convert_element_type_37, convert_element_type_38, erf_2, mul_27, mul_28, mul_29
# l__self___blocks_0_local_mp_bn => add_22, convert_element_type_41, mul_31, mul_32, sub_6
triton_poi_fused__native_batch_norm_legit_no_training_gelu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_gelu_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_gelu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''')
