

# Original file: ./hrnet_w18___60.0/hrnet_w18___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/jt/cjtiwan2vc54vaghlns5dyg4nnjjnixewstuy3g5wdbrgksdc2g4.py
# Source Nodes: [add_5, l__mod___stage3_0_fuse_act_1, l__mod___stage3_0_fuse_layers_1_2_1, l__mod___stage3_0_fuse_layers_1_2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten._unsafe_index, aten.add, aten.relu]
# add_5 => add_165
# l__mod___stage3_0_fuse_act_1 => relu_60
# l__mod___stage3_0_fuse_layers_1_2_1 => add_162, mul_202, mul_203, sub_63
# l__mod___stage3_0_fuse_layers_1_2_2 => _unsafe_index_3, convert_element_type_215
triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x2 = (xindex // 1008) % 28
    x1 = (xindex // 36) % 28
    x0 = xindex % 36
    x3 = (xindex // 28224)
    tmp0 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
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
    tmp16 = tl.load(in_ptr0 + (x0 + (36*tmp15) + (504*tmp9) + (7056*x3)), None).to(tl.float32)
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
    tl.store(in_out_ptr0 + (x4), tmp30, None)
''')
