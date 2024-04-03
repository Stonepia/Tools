

# Original file: ./phlippe_densenet___60.0/phlippe_densenet___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/b7/cb7f6zyv3q7fptxzr42r3aidloruimxuvdsekxrf5lid7ybdp5mz.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4___block___0___net_0, getattr_getattr_l__mod___blocks___4___block___0___net_1, getattr_l__mod___blocks___3___transition_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.relu]
# getattr_getattr_l__mod___blocks___4___block___0___net_0 => add_53, mul_79, mul_80, sub_26
# getattr_getattr_l__mod___blocks___4___block___0___net_1 => relu_26
# getattr_l__mod___blocks___3___transition_3 => avg_pool2d_1
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_27', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_27', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 80
    x1 = (xindex // 80) % 8
    x2 = (xindex // 640)
    x3 = xindex
    x4 = (xindex // 80)
    tmp0 = tl.load(in_ptr0 + (x0 + (160*x1) + (2560*x2)), None)
    tmp1 = tl.load(in_ptr0 + (80 + x0 + (160*x1) + (2560*x2)), None)
    tmp3 = tl.load(in_ptr0 + (1280 + x0 + (160*x1) + (2560*x2)), None)
    tmp5 = tl.load(in_ptr0 + (1360 + x0 + (160*x1) + (2560*x2)), None)
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14 + tmp9
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr1 + (x0 + (96*x4)), tmp8, None)
''')
