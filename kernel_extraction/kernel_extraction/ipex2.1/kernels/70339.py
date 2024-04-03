

# Original file: ./maml__21_backward_64.2/maml__21_backward_64.2_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/dg/cdggekorypdzjy4j26vc4wtdrztzuuv5hm5q33e5xb25j2ula5gf.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_1
triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 5)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp14 = tl.load(in_ptr5 + (x2), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp9 = tmp6 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.exp(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp13 - tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp0 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''')