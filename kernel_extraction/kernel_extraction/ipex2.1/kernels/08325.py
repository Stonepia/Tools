

# Original file: ./MegatronBertForCausalLM__0_backward_351.1/MegatronBertForCausalLM__0_backward_351.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/2l/c2lavw3nbxno45wmhj4pr3b2mpsajbe5rhnp5ghlyr5d3o6oogxs.py
# Source Nodes: [cross_entropy], Original ATen: [aten.embedding_dense_backward, aten.native_dropout_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_3
triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_24 = async_compile.triton('triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_24', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_24', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_native_dropout_backward_nll_loss_forward_sum_24(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp9 = tl.load(in_ptr1 + (524288 + x2), None)
    tmp10 = tl.load(in_ptr2 + (524288 + x2), None)
    tmp15 = tl.load(in_ptr1 + (1048576 + x2), None)
    tmp16 = tl.load(in_ptr2 + (1048576 + x2), None)
    tmp21 = tl.load(in_ptr1 + (1572864 + x2), None)
    tmp22 = tl.load(in_ptr2 + (1572864 + x2), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 1.1111111111111112
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 * tmp7
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp6
    tmp13 = tmp9 * tmp12
    tmp14 = tmp8 + tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 * tmp6
    tmp19 = tmp15 * tmp18
    tmp20 = tmp14 + tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 * tmp6
    tmp25 = tmp21 * tmp24
    tmp26 = tmp20 + tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp2, tmp27, tmp26)
    tmp29 = tl.where(tmp0 < 0, tmp0 + 512, tmp0)
    tl.atomic_add(out_ptr1 + (x0 + (1024*tmp29)), tmp28, None)
''')
