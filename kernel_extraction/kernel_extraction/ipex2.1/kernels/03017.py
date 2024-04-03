

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/kj/ckjomn62uqcomhmaccon2hsp6vnzv3boirtlwec7pbovrt3wl7bu.py
# Source Nodes: [iadd_1, nan_to_num, softmax, type_as_2], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.nan_to_num]
# iadd_1 => add_4
# nan_to_num => convert_element_type_12, eq, eq_1, full_default_2, full_default_3, full_default_4, isnan, where_1, where_2, where_3
# softmax => amax, div, exp, sub_2, sum_1
# type_as_2 => convert_element_type_16
triton_poi_fused__softmax__to_copy_add_nan_to_num_5 = async_compile.triton('triton_poi_fused__softmax__to_copy_add_nan_to_num_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_add_nan_to_num_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_add_nan_to_num_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
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
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp18 / tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
''')
