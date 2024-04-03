

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/r6/cr6pah7xyuxejqahxlzjptkxbk6fzrogpthncgdqjrhgh7rux5m7.py
# Source Nodes: [iadd_2, nan_to_num, nan_to_num_1, softmax_1], Original ATen: [aten._softmax, aten.add, aten.nan_to_num]
# iadd_2 => add_11
# nan_to_num => full_default_2, full_default_3, full_default_4
# nan_to_num_1 => convert_element_type_33, eq_2, eq_3, isnan_1, where_4, where_5, where_6
# softmax_1 => amax_1, sub_5
triton_poi_fused__softmax_add_nan_to_num_10 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_10', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr0 + (1 + (2*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 == tmp2
    tmp20 = tmp18 == tmp4
    tmp21 = libdevice.isnan(tmp17).to(tl.int1)
    tmp22 = tl.where(tmp21, tmp7, tmp17)
    tmp23 = tl.where(tmp20, tmp9, tmp22)
    tmp24 = tl.where(tmp19, tmp11, tmp23)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 + tmp15
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 == tmp2
    tmp30 = tmp28 == tmp4
    tmp31 = libdevice.isnan(tmp27).to(tl.int1)
    tmp32 = tl.where(tmp31, tmp7, tmp27)
    tmp33 = tl.where(tmp30, tmp9, tmp32)
    tmp34 = tl.where(tmp29, tmp11, tmp33)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 + tmp15
    tmp37 = triton_helpers.maximum(tmp26, tmp36)
    tmp38 = tmp16 - tmp37
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''')
