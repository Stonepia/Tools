

# Original file: ./yolov3___60.0/yolov3___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/je/cjenlhwhmz7ebronrbjjyopdvharjwbvtxlytjiysv4tgslfnjig.py
# Source Nodes: [add_19, l__mod___module_list_64_activation, l__mod___module_list_64_batch_norm2d], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.leaky_relu]
# add_19 => add_111
# l__mod___module_list_64_activation => convert_element_type_229, gt_45, mul_183, where_45
# l__mod___module_list_64_batch_norm2d => add_110, mul_181, mul_182, sub_45
triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_leaky_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 + tmp10
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.1
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 + tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''')
