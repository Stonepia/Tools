

# Original file: ./phlippe_densenet___60.0/phlippe_densenet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/yb/cybolmije3ofolz577mhekro6wiofrvpii5nl35qr2zzhs5gjtru.py
# Source Nodes: [cat_45, getattr_getattr_l__self___blocks___0___block___2___net_0, getattr_getattr_l__self___blocks___0___block___2___net_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_45 => cat_2
# getattr_getattr_l__self___blocks___0___block___2___net_0 => add_9, convert_element_type_21, mul_13, mul_14, sub_4
# getattr_getattr_l__self___blocks___0___block___2___net_1 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(0, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x0 + (80*x1)), tmp0, None)
''')