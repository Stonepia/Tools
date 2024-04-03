

# Original file: ./yolov3___60.0/yolov3___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/eu/ceueuit23ugrg365c3y2ytkljdrvjnrquz256ubwtoa7xunbaooq.py
# Source Nodes: [add_3, l__self___module_list_14_activation, l__self___module_list_14_batch_norm2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# add_3 => add_27
# l__self___module_list_14_activation => convert_element_type_72, gt_11, mul_47, where_11
# l__self___module_list_14_batch_norm2d => add_26, mul_45, mul_46, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = 0.0
    tmp10 = tmp8 > tmp9
    tmp11 = 0.1
    tmp12 = tmp8 * tmp11
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''')
