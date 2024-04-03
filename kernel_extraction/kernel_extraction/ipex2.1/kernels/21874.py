

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/k2/ck2cu7wxjo6mcb4e2zsd7tgtdim3pnxrqrtgzqnvxun3su6szq5y.py
# Source Nodes: [add, l__mod___stage2_0_fuse_act, l__mod___stage2_0_fuse_layers_0_1_1, l__mod___stage2_0_fuse_layers_0_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten._unsafe_index, aten.add, aten.relu]
# add => add_82
# l__mod___stage2_0_fuse_act => relu_32
# l__mod___stage2_0_fuse_layers_0_1_1 => add_79, mul_100, mul_101, sub_33
# l__mod___stage2_0_fuse_layers_0_1_2 => _unsafe_index, convert_element_type_107
triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 56
    x1 = (xindex // 18) % 56
    x0 = xindex % 18
    x3 = (xindex // 56448)
    tmp0 = tl.load(in_ptr0 + (x4), None).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = x2
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x1
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp3
    tmp13 = tmp12 + tmp5
    tmp14 = tmp13 * tmp7
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr1 + (x0 + (18*tmp15) + (504*tmp9) + (14112*x3)), None).to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 - tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp0 + tmp28
    tmp30 = triton_helpers.maximum(0, tmp29)
    tl.store(out_ptr0 + (x4), tmp30, None)
''')
