

# Original file: ./phlippe_densenet___60.0/phlippe_densenet___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/n7/cn73afvwsdisboheuqoz73pnrf2bawpqutwdhtts7kzjogkxj4em.py
# Source Nodes: [getattr_getattr_l__mod___blocks___6___block___0___net_0, getattr_getattr_l__mod___blocks___6___block___0___net_1, getattr_l__mod___blocks___5___transition_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.relu]
# getattr_getattr_l__mod___blocks___6___block___0___net_0 => add_79, convert_element_type_119, mul_118, mul_119, sub_39
# getattr_getattr_l__mod___blocks___6___block___0___net_1 => relu_39
# getattr_l__mod___blocks___5___transition_3 => avg_pool2d_2
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_40', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_40', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 180224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 88
    x1 = (xindex // 88) % 4
    x2 = (xindex // 352)
    x3 = xindex
    x4 = (xindex // 88)
    tmp0 = tl.load(in_ptr0 + (x0 + (176*x1) + (1408*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (88 + x0 + (176*x1) + (1408*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (704 + x0 + (176*x1) + (1408*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (792 + x0 + (176*x1) + (1408*x2)), None).to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = triton_helpers.maximum(0, tmp20)
    tl.store(out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (x0 + (104*x4)), tmp8, None)
''')
