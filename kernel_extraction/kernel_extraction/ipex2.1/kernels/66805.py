

# Original file: ./hf_T5_generate__23_inference_63.3/hf_T5_generate__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/64/c64b3btwfi4tqwuzamx2l6iath5smtysvekellg32xts35ucugtq.py
# Source Nodes: [float_2, softmax, type_as], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_11
# softmax => div_2, exp, sub_3
# type_as => convert_element_type_12
triton_poi_fused__softmax__to_copy_4 = async_compile.triton('triton_poi_fused__softmax__to_copy_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = (-1)*(tl.minimum(0, (-1) + x0, tl.PropagateNan.NONE))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 16.0
    tmp5 = tmp3 / tmp4
    tmp6 = tl.log(tmp5)
    tmp7 = 2.0794415416798357
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 16, tl.int64)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.full([1], 31, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.full([1], True, tl.int1)
    tmp16 = tl.where(tmp15, tmp2, tmp14)
    tmp17 = tl.full([1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
    # tl.device_assert(0 <= tmp19, "index out of bounds: 0 <= tmp19")
    tmp20 = tl.load(in_ptr0 + (x1 + (8*tmp19)), xmask)
    tmp21 = tmp1 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp23 - tmp24
    tmp26 = tl.exp(tmp25)
    tmp28 = tmp26 / tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''')
