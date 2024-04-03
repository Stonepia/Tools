

# Original file: ./hf_T5_generate__23_inference_63.3/hf_T5_generate__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/5d/c5dbvlxhkwutk7pqkttlpotzeskco7dn3l7amolocmo5fqxfewdc.py
# Source Nodes: [float_2, softmax], Original ATen: [aten._softmax, aten._to_copy]
# float_2 => convert_element_type_11
# softmax => amax, exp, sub_3, sum_1
triton_poi_fused__softmax__to_copy_3 = async_compile.triton('triton_poi_fused__softmax__to_copy_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask).to(tl.float32)
    tmp24 = tl.load(in_ptr0 + (1 + (2*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = 16.0
    tmp4 = tmp2 / tmp3
    tmp5 = tl.log(tmp4)
    tmp6 = 2.0794415416798357
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7 * tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 16, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tl.full([1], True, tl.int1)
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tmp17 = tl.full([1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
    tmp20 = tl.load(in_ptr1 + (x0 + (8*tmp19)), xmask)
    tmp21 = tmp1 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 0.0
    tmp27 = tmp26 / tmp3
    tmp28 = tl.log(tmp27)
    tmp29 = tmp28 / tmp6
    tmp30 = tmp29 * tmp3
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tmp31 + tmp10
    tmp33 = triton_helpers.minimum(tmp32, tmp12)
    tmp34 = tl.where(tmp14, tmp17, tmp33)
    tmp35 = tmp34 + tmp17
    tmp36 = tl.where(tmp35 < 0, tmp35 + 32, tmp35)
    # tl.device_assert(0 <= tmp36, "index out of bounds: 0 <= tmp36")
    tmp37 = tl.load(in_ptr1 + (x0 + (8*tmp36)), xmask)
    tmp38 = tmp25 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = triton_helpers.maximum(tmp23, tmp40)
    tmp42 = tmp23 - tmp41
    tmp43 = tl.exp(tmp42)
    tmp44 = tmp40 - tmp41
    tmp45 = tl.exp(tmp44)
    tmp46 = tmp43 + tmp45
    tl.store(out_ptr0 + (x0), tmp41, xmask)
    tl.store(out_ptr1 + (x0), tmp46, xmask)
''')
