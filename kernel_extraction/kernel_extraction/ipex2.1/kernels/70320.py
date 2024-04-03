

# Original file: ./maml__21_backward_64.2/maml__21_backward_64.2_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/3j/c3jmmpintyszjb4x4jnqkidaxq2jxnwjp7ffqekilm55npvnd5mt.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_1
triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + (1 + (5*x0)), xmask)
    tmp15 = tl.load(in_ptr0 + (2 + (5*x0)), xmask)
    tmp18 = tl.load(in_ptr0 + (3 + (5*x0)), xmask)
    tmp21 = tl.load(in_ptr0 + (4 + (5*x0)), xmask)
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp8 = tmp5 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp3, tmp8, tmp9)
    tmp11 = tmp0 * tmp10
    tmp13 = tmp12 * tmp10
    tmp14 = tmp11 + tmp13
    tmp16 = tmp15 * tmp10
    tmp17 = tmp14 + tmp16
    tmp19 = tmp18 * tmp10
    tmp20 = tmp17 + tmp19
    tmp22 = tmp21 * tmp10
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr0 + (x0), tmp23, xmask)
''')
