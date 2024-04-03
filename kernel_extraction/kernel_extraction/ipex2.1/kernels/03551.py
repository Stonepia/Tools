

# Original file: ./T5Small__0_backward_171.1/T5Small__0_backward_171.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/do/cdodsa2p5v7cbrgwc34rcliknhe7qdo735rt7lp7bxm44ndrhiqv.py
# Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.embedding_dense_backward, aten.sum]

triton_poi_fused_add_clone_embedding_dense_backward_sum_23 = async_compile.triton('triton_poi_fused_add_clone_embedding_dense_backward_sum_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_embedding_dense_backward_sum_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_embedding_dense_backward_sum_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    x3 = xindex % 1048576
    x4 = (xindex // 1048576)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (x2), None)
    tmp10 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8388608 + x2), None).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (8388608 + x2), None).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (8388608 + x2), None)
    tmp22 = tl.load(in_ptr3 + (8388608 + x2), None)
    tmp24 = tl.load(in_ptr4 + (8192 + x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (16777216 + x2), None).to(tl.float32)
    tmp31 = tl.load(in_ptr1 + (16777216 + x2), None).to(tl.float32)
    tmp32 = tl.load(in_ptr2 + (16777216 + x2), None)
    tmp37 = tl.load(in_ptr3 + (16777216 + x2), None)
    tmp39 = tl.load(in_ptr4 + (16384 + x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (25165824 + x2), None).to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (25165824 + x2), None).to(tl.float32)
    tmp47 = tl.load(in_ptr2 + (25165824 + x2), None)
    tmp52 = tl.load(in_ptr3 + (25165824 + x2), None)
    tmp54 = tl.load(in_ptr4 + (24576 + x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp8 * tmp10
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp0 + tmp13
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp4
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp25 = tmp22 * tmp24
    tmp26 = tmp23 - tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp15 + tmp27
    tmp29 = tmp14 + tmp28
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp4
    tmp35 = tmp31 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp38 = tmp36 * tmp37
    tmp40 = tmp37 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp30 + tmp42
    tmp44 = tmp29 + tmp43
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp4
    tmp50 = tmp46 * tmp49
    tmp51 = tmp50.to(tl.float32)
    tmp53 = tmp51 * tmp52
    tmp55 = tmp52 * tmp54
    tmp56 = tmp53 - tmp55
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp45 + tmp57
    tmp59 = tmp44 + tmp58
    tmp61 = tl.where(tmp60 < 0, tmp60 + 32, tmp60)
    tmp62 = tmp59.to(tl.float32)
    tmp63 = tl.full([1], False, tl.int1)
    tmp64 = 0.0
    tmp65 = tl.where(tmp63, tmp64, tmp62)
    tl.atomic_add(out_ptr1 + (x4 + (8*tmp61)), tmp65, None)
''')
