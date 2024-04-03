

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/qc/cqckip5ryz5u2fvdyxvsge7iloc6f6yrqvzbrycphedphc63vkcq.py
# Source Nodes: [iadd_7, nan_to_num, nan_to_num_6, softmax_6, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_7 => add_46
# nan_to_num => full_default_3, full_default_4
# nan_to_num_6 => eq_12, eq_13, isnan_6, where_19, where_20, where_21
# softmax_6 => div_6, exp_6, sub_20
# triu => full_default_1
triton_poi_fused__softmax_add_nan_to_num_triu_34 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_triu_34', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_triu_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_triu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 7)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp12 = tl.load(in_ptr0 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = float("inf")
    tmp2 = tmp0 == tmp1
    tmp3 = float("-inf")
    tmp4 = tmp0 == tmp3
    tmp5 = libdevice.isnan(tmp0).to(tl.int1)
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = -3.4028234663852886e+38
    tmp9 = tl.where(tmp4, tmp8, tmp7)
    tmp10 = 3.4028234663852886e+38
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp14 = tmp11 + tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tl.exp(tmp16)
    tmp19 = tmp17 / tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''')
