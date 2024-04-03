

# Original file: ./phlippe_densenet___60.0/phlippe_densenet___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/x5/cx5gzs7ya2yeltuvzm6kmk5foplakk7hc6lwzmvhyumhzeffyxdj.py
# Source Nodes: [cat_36, getattr_getattr_l__mod___blocks___2___block___5___net_0, getattr_getattr_l__mod___blocks___2___block___5___net_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_36 => cat_11
# getattr_getattr_l__mod___blocks___2___block___5___net_0 => add_47, mul_70, mul_71, sub_23
# getattr_getattr_l__mod___blocks___2___block___5___net_1 => relu_23
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 + tmp1
    tmp8 = triton_helpers.maximum(0, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x0 + (160*x1)), tmp0, None)
''')
