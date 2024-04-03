

# Original file: ./LayoutLMForSequenceClassification__0_forward_133.0/LayoutLMForSequenceClassification__0_forward_133.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/aq/caq3nf4n3uh7jzsntyi6c5gzpkcqspamh3wuggi6lzzff5ypybrb.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# cross_entropy => amax_12, convert_element_type_100, convert_element_type_99, exp_12, log, sub_40, sub_41, sum_13
triton_poi_fused__log_softmax_9 = async_compile.triton('triton_poi_fused__log_softmax_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (1 + (2*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = triton_helpers.maximum(tmp3, tmp5)
    tmp7 = tmp1 - tmp6
    tmp8 = tmp3 - tmp6
    tmp9 = tl.exp(tmp8)
    tmp10 = tmp5 - tmp6
    tmp11 = tl.exp(tmp10)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.log(tmp12)
    tmp14 = tmp7 - tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''')
