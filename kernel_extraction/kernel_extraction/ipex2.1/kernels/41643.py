

# Original file: ./maml__21_forward_62.1/maml__21_forward_62.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/vb/cvbct666b6fofjtprcehxjes3v4pkh7ag7hdlcuplz4ywnpyqvkp.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# cross_entropy => amax, convert_element_type_16, exp, log, sub_4, sum_1
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

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_12(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (1 + (5*x0)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2 + (5*x0)), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (3 + (5*x0)), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (4 + (5*x0)), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp1 - tmp13
    tmp15 = tl.exp(tmp14)
    tmp16 = tmp3 - tmp13
    tmp17 = tl.exp(tmp16)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp6 - tmp13
    tmp20 = tl.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp13
    tmp23 = tl.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp12 - tmp13
    tmp26 = tl.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tl.log(tmp27)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
''')
