

# Original file: ./GPT2ForSequenceClassification__0_forward_133.0/GPT2ForSequenceClassification__0_forward_133.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/wq/cwqtlpdu35njvs62li4t4vmsnz57yjudyybhqd2beynvuhkgujop.py
# Source Nodes: [add, l__mod___transformer_drop, l__mod___transformer_h_0_ln_1, l__mod___transformer_wpe, l__mod___transformer_wte], Original ATen: [aten.add, aten.embedding, aten.native_dropout, aten.native_layer_norm]
# add => add
# l__mod___transformer_drop => gt, mul, mul_1
# l__mod___transformer_h_0_ln_1 => add_1, add_2, convert_element_type, convert_element_type_1, mul_2, mul_3, rsqrt, sub, var_mean
# l__mod___transformer_wpe => embedding_1
# l__mod___transformer_wte => embedding
triton_per_fused_add_embedding_native_dropout_native_layer_norm_1 = async_compile.triton('triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*i1', 8: '*bf16', 9: '*fp32', 10: '*bf16', 11: 'i32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_dropout_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_embedding_native_dropout_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, out_ptr3, out_ptr4, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 1024
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r1 + (768*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (768*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tl.where(tmp7 < 0, tmp7 + 50257, tmp7)
    # tl.device_assert((0 <= tmp8) & (tmp8 < 50257), "index out of bounds: 0 <= tmp8 < 50257")
    tmp9 = tl.load(in_ptr2 + (r1 + (768*tmp8)), rmask, other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = 1.1111111111111112
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 / tmp24
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = 768.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp15 - tmp25
    tmp38 = tmp37 * tmp36
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp5, rmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp36, None)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp45, rmask)
    tl.store(out_ptr3 + (x0), tmp25, None)
''')
