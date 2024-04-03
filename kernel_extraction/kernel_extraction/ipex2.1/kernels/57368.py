

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/cd/ccdneplvlepcb7byhfj47lug2bkwkvu7o3n7gpqb65waupg5peft.py
# Source Nodes: [add, add_1, add_2, add_3, forward, forward_1, mean, mul, std, sub, truediv], Original ATen: [aten.add, aten.div, aten.embedding, aten.mean, aten.mul, aten.std, aten.sub]
# add => add
# add_1 => add_1
# add_2 => add_2
# add_3 => add_3
# forward => embedding
# forward_1 => embedding_1
# mean => mean
# mul => mul
# std => convert_element_type, convert_element_type_1, sqrt, var
# sub => sub
# truediv => div
triton_per_fused_add_div_embedding_mean_mul_std_sub_1 = async_compile.triton('triton_per_fused_add_div_embedding_mean_mul_std_sub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*i64', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_embedding_mean_mul_std_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_embedding_mean_mul_std_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 20005, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 20005), "index out of bounds: 0 <= tmp1 < 20005")
    tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp6 = tl.where(tmp5 < 0, tmp5 + 3, tmp5)
    # tl.device_assert((0 <= tmp6) & (tmp6 < 3), "index out of bounds: 0 <= tmp6 < 3")
    tmp7 = tl.load(in_ptr4 + (r2 + (768*tmp6)), rmask, other=0.0).to(tl.float32)
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tl.full([1], 768, tl.int32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 / tmp20
    tmp22 = tmp10 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp13 / tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp8 - tmp31
    tmp33 = tmp28 * tmp32
    tmp34 = 767.0
    tmp35 = tmp27 / tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 1e-06
    tmp39 = tmp37 + tmp38
    tmp40 = tmp33 / tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp8, rmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp42, rmask)
''')
