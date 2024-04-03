

# Original file: ./sam___60.0/sam___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/4x/c4xjfziiysy4afctgiakv27eeprhwmkckmooa6uyyd2xaujhbcgu.py
# Source Nodes: [softmax_34, truediv_7], Original ATen: [aten._softmax, aten.div]
# softmax_34 => amax_34, convert_element_type_631, exp_34, sub_173, sum_35
# truediv_7 => div_41
triton_poi_fused__softmax_div_54 = async_compile.triton('triton_poi_fused__softmax_div_54', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_54', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_div_54(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (1 + (5*x0)), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (2 + (5*x0)), None).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (3 + (5*x0)), None).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (4 + (5*x0)), None).to(tl.float32)
    tmp1 = 4.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4 / tmp1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp9 = tmp8 / tmp1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = triton_helpers.maximum(tmp7, tmp10)
    tmp13 = tmp12 / tmp1
    tmp14 = tmp13.to(tl.float32)
    tmp15 = triton_helpers.maximum(tmp11, tmp14)
    tmp17 = tmp16 / tmp1
    tmp18 = tmp17.to(tl.float32)
    tmp19 = triton_helpers.maximum(tmp15, tmp18)
    tmp20 = tmp3 - tmp19
    tmp21 = tl.exp(tmp20)
    tmp22 = tmp6 - tmp19
    tmp23 = tl.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp10 - tmp19
    tmp26 = tl.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp14 - tmp19
    tmp29 = tl.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tmp18 - tmp19
    tmp32 = tl.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr0 + (x0), tmp19, None)
    tl.store(out_ptr1 + (x0), tmp33, None)
''')
