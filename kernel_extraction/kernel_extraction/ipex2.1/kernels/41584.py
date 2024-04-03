

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6h/c6hzv2y7ltazgjebuw57ikzv475ikt4jlo4pzx643wuvzjykmumu.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# cross_entropy => amax, exp, log, sub_4, sum_1
triton_poi_fused__log_softmax_12 = async_compile.triton('triton_poi_fused__log_softmax_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_12(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + (5*x0)), xmask)
    tmp3 = tl.load(in_ptr0 + (2 + (5*x0)), xmask)
    tmp5 = tl.load(in_ptr0 + (3 + (5*x0)), xmask)
    tmp7 = tl.load(in_ptr0 + (4 + (5*x0)), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp0 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tmp1 - tmp8
    tmp12 = tl.exp(tmp11)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tl.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tl.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp7 - tmp8
    tmp21 = tl.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tl.log(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''')
