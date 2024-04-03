

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/d5/cd5rwf7gniy3bowi3e7zrmje5bo3x3e3v5aqm3yaqjfrjtgbwxdl.py
# Source Nodes: [iadd_3, nan_to_num, nan_to_num_2, softmax_2, type_as_4], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_3 => add_18
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_2 => convert_element_type_22, eq_4, eq_5, isnan_2, where_7, where_8, where_9
# softmax_2 => convert_element_type_24, div_2, exp_2, sub_8
# type_as_4 => convert_element_type_25
triton_poi_fused__softmax__to_copy_add_nan_to_num_17 = async_compile.triton('triton_poi_fused__softmax__to_copy_add_nan_to_num_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[64], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_nan_to_num_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_add_nan_to_num_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 3)
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
    tmp9 = -65504.0
    tmp10 = tl.where(tmp5, tmp9, tmp8)
    tmp11 = 65504.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp15 = tmp12 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp19 = tl.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp22, xmask)
''')
