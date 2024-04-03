

# Original file: ./visformer_small___60.0/visformer_small___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/ms/cms7i7dww6uvklslv5dxzxdmkk26hup6gs5ymoo4z7ceuc72kpmo.py
# Source Nodes: [add, add_1, add_2, add_3, getattr_l__self___stage1___3___mlp_conv1, getattr_l__self___stage1___3___norm2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._to_copy, aten.add]
# add => add_4
# add_1 => add_9
# add_2 => add_14
# add_3 => add_19
# getattr_l__self___stage1___3___mlp_conv1 => convert_element_type_42
# getattr_l__self___stage1___3___norm2 => add_21, mul_34, mul_35, sub_5
triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*bf16', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__to_copy_add_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 192
    x1 = (xindex // 192) % 784
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1 + (784*x0)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp12 - tmp13
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''')
