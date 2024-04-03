

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/eu/ceuw44jydbe46n5v3h47corjkavcdufr2gtu5wcomtfopr77lzw3.py
# Source Nodes: [iadd_5, nan_to_num, nan_to_num_4, softmax_4, type_as_6], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_5 => add_32
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_4 => convert_element_type_38, eq_8, eq_9, isnan_4, where_13, where_14, where_15
# softmax_4 => convert_element_type_40, div_4, exp_4, sub_14
# type_as_6 => convert_element_type_41
triton_poi_fused__softmax__to_copy_add_nan_to_num_26 = async_compile.triton('triton_poi_fused__softmax__to_copy_add_nan_to_num_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_nan_to_num_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_add_nan_to_num_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 5)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp17 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = float("inf")
    tmp3 = tmp1 == tmp2
    tmp4 = float("-inf")
    tmp5 = tmp1 == tmp4
    tmp6 = libdevice.isnan(tmp0).to(tl.int1)
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp0)
    tmp9 = -3.3895313892515355e+38
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 3.3895313892515355e+38
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp22, xmask)
''')
