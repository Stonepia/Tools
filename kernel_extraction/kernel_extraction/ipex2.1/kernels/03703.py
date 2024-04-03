

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/5o/c5ooeivwuoint7avlp7sd6whoit6vlukzgav253mgrd5rsianfrw.py
# Source Nodes: [cross_entropy], Original ATen: [aten.add, aten.clone, aten.embedding_dense_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_7
triton_poi_fused_add_clone_embedding_dense_backward_nll_loss_forward_sum_23 = async_compile.triton('triton_poi_fused_add_clone_embedding_dense_backward_nll_loss_forward_sum_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_embedding_dense_backward_nll_loss_forward_sum_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_embedding_dense_backward_nll_loss_forward_sum_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    x3 = xindex % 1048576
    x4 = (xindex // 1048576)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (8388608 + x2), None)
    tmp14 = tl.load(in_ptr1 + (8388608 + x2), None)
    tmp15 = tl.load(in_ptr2 + (8388608 + x2), None)
    tmp19 = tl.load(in_ptr3 + (8388608 + x2), None)
    tmp21 = tl.load(in_ptr4 + (8192 + x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (16777216 + x2), None)
    tmp27 = tl.load(in_ptr1 + (16777216 + x2), None)
    tmp28 = tl.load(in_ptr2 + (16777216 + x2), None)
    tmp32 = tl.load(in_ptr3 + (16777216 + x2), None)
    tmp34 = tl.load(in_ptr4 + (16384 + x1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (25165824 + x2), None)
    tmp40 = tl.load(in_ptr1 + (25165824 + x2), None)
    tmp41 = tl.load(in_ptr2 + (25165824 + x2), None)
    tmp45 = tl.load(in_ptr3 + (25165824 + x2), None)
    tmp47 = tl.load(in_ptr4 + (24576 + x1), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp7 * tmp9
    tmp11 = tmp8 - tmp10
    tmp12 = tmp0 + tmp11
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp4
    tmp18 = tmp14 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp19 * tmp21
    tmp23 = tmp20 - tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 + tmp24
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp4
    tmp31 = tmp27 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp32 * tmp34
    tmp36 = tmp33 - tmp35
    tmp37 = tmp26 + tmp36
    tmp38 = tmp25 + tmp37
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp4
    tmp44 = tmp40 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp45 * tmp47
    tmp49 = tmp46 - tmp48
    tmp50 = tmp39 + tmp49
    tmp51 = tmp38 + tmp50
    tmp53 = tl.where(tmp52 < 0, tmp52 + 32, tmp52)
    tmp54 = tl.full([1], False, tl.int1)
    tmp55 = 0.0
    tmp56 = tl.where(tmp54, tmp55, tmp51)
    tl.atomic_add(out_ptr1 + (x4 + (8*tmp53)), tmp56, None)
''')
