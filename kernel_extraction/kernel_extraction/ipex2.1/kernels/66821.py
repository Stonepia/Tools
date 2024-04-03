

# Original file: ./hf_T5_generate__23_inference_63.3/hf_T5_generate__23_inference_63.3_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/2h/c2hrkmyptl4kg6jxvh5eusr6heeicez4mhoafajfayd7sdg6hi2o.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => div_2, exp, sub_3
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp21 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = (-1)*(tl.minimum(0, (-1) + x0, tl.PropagateNan.NONE))
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 16.0
    tmp4 = tmp2 / tmp3
    tmp5 = tl.log(tmp4)
    tmp6 = 2.0794415416798357
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([1], 16, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tl.full([1], True, tl.int1)
    tmp15 = tl.where(tmp14, tmp1, tmp13)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.where(tmp17 < 0, tmp17 + 32, tmp17)
    # tl.device_assert(0 <= tmp18, "index out of bounds: 0 <= tmp18")
    tmp19 = tl.load(in_ptr0 + (x1 + (8*tmp18)), xmask)
    tmp20 = tmp0 + tmp19
    tmp22 = tmp20 - tmp21
    tmp23 = tl.exp(tmp22)
    tmp25 = tmp23 / tmp24
    tl.store(in_out_ptr0 + (x2), tmp25, xmask)
''')
