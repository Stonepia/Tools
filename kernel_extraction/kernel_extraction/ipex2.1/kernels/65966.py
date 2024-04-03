

# Original file: ./visformer_small___60.0/visformer_small___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/nd/cndwjxoikfpprglibbbegqces5ypcbqgfjkkffvsi44o5xis3ics.py
# Source Nodes: [add_21, add_22, add_23, getattr_l__mod___stage3___3___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_21 => add_87
# add_22 => add_90
# add_23 => add_94
# getattr_l__mod___stage3___3___norm1 => add_96, mul_146, mul_147, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_add_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_36', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_36', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + (x2), tmp14, None)
''')
