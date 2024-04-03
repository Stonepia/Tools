

# Original file: ./hf_T5_generate__23_inference_63.3/hf_T5_generate__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2o/c2oooxvvqbtoudseyzlhqtgdt53uttn4otuwgekkzx7uqhonwvhz.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, exp, sub_3, sum_1
triton_poi_fused__softmax_3 = async_compile.triton('triton_poi_fused__softmax_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask)
    tmp21 = tl.load(in_ptr0 + (1 + (2*x0)), xmask)
    tmp1 = 1.0
    tmp2 = 16.0
    tmp3 = tmp1 / tmp2
    tmp4 = tl.log(tmp3)
    tmp5 = 2.0794415416798357
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 31, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = tl.full([1], True, tl.int1)
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tl.where(tmp13, tmp14, tmp12)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
    tmp19 = tl.load(in_ptr1 + (x0 + (8*tmp18)), xmask)
    tmp20 = tmp0 + tmp19
    tmp22 = 0.0
    tmp23 = tmp22 / tmp2
    tmp24 = tl.log(tmp23)
    tmp25 = tmp24 / tmp5
    tmp26 = tmp25 * tmp2
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tmp27 + tmp9
    tmp29 = triton_helpers.minimum(tmp28, tmp11)
    tmp30 = tl.where(tmp13, tmp16, tmp29)
    tmp31 = tmp30 + tmp16
    tmp32 = tl.where(tmp31 < 0, tmp31 + 32, tmp31)
    # tl.device_assert(0 <= tmp32, "index out of bounds: 0 <= tmp32")
    tmp33 = tl.load(in_ptr1 + (x0 + (8*tmp32)), xmask)
    tmp34 = tmp21 + tmp33
    tmp35 = triton_helpers.maximum(tmp20, tmp34)
    tmp36 = tmp20 - tmp35
    tmp37 = tl.exp(tmp36)
    tmp38 = tmp34 - tmp35
    tmp39 = tl.exp(tmp38)
    tmp40 = tmp37 + tmp39
    tl.store(out_ptr0 + (x0), tmp35, xmask)
    tl.store(out_ptr1 + (x0), tmp40, xmask)
''')
