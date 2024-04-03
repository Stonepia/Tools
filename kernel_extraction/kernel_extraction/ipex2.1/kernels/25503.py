

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/rb/crbgks5mf4co2kuggkgelvaswbhtiveus5zktfq5nq6pcek35kto.py
# Source Nodes: [softmax_32, truediv_5], Original ATen: [aten._softmax, aten.div]
# softmax_32 => amax_32, convert_element_type_322, convert_element_type_323, div_38, exp_32, sub_168, sum_33
# truediv_5 => div_37
triton_poi_fused__softmax_div_28 = async_compile.triton('triton_poi_fused__softmax_div_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256], filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax_div_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 5)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 5.656854249492381
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp6 = tl.exp(tmp5)
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp9, xmask)
''')
