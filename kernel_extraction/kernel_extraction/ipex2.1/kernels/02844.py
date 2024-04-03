

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/q7/cq7jurmqt4olfoo2oiubwzy4yy2ygsf4bqgcllven7rj4ba5tg46.py
# Source Nodes: [iadd_1, nan_to_num, softmax, triu], Original ATen: [aten._softmax, aten.add, aten.nan_to_num, aten.triu]
# iadd_1 => add_4
# nan_to_num => eq, eq_1, full_default_3, full_default_4, isnan, where_1, where_2, where_3
# softmax => amax, div, exp, sub_2, sum_1
# triu => full_default_1
triton_poi_fused__softmax_add_nan_to_num_triu_5 = async_compile.triton('triton_poi_fused__softmax_add_nan_to_num_triu_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_nan_to_num_triu_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_add_nan_to_num_triu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp12 = tl.load(in_ptr0 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
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
    tmp15 = tmp14 - tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = tmp16 / tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)
''')
