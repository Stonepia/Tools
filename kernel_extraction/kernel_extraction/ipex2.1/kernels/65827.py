

# Original file: ./visformer_small___60.0/visformer_small___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/ub/cubqwvk2fofn3byy4j7rov6waxkfjwwswni6bibhlradc4xrvzku.py
# Source Nodes: [add_10, add_8, add_9, getattr_l__self___stage2___1___attn_qkv, getattr_l__self___stage2___1___norm1], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten.add]
# add_10 => add_49
# add_8 => add_42
# add_9 => add_45
# getattr_l__self___stage2___1___attn_qkv => convert_element_type_102
# getattr_l__self___stage2___1___norm1 => add_51, mul_83, mul_84, sub_13
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9633792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 384
    x1 = (xindex // 384) % 196
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1 + (196*x0)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp18, None)
''')
