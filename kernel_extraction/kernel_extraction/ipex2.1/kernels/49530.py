

# Original file: ./xcit_large_24_p8_224___60.0/xcit_large_24_p8_224___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/lt/cltpt7eoj4of5hsj3vaoauwvoxnd5w455sidca3lvryqzojilbhu.py
# Source Nodes: [l__self___patch_embed_proj_2_1, l__self___patch_embed_proj_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.gelu]
# l__self___patch_embed_proj_2_1 => add_4, mul_7, mul_8, sub_1
# l__self___patch_embed_proj_3 => add_5, convert_element_type_12, erf_1, mul_10, mul_11, mul_9
triton_poi_fused__native_batch_norm_legit_no_training_gelu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_gelu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_gelu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_gelu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
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
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7071067811865476
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp18, None)
''')
