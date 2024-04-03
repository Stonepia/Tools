

# Original file: ./T5ForConditionalGeneration__0_forward_169.0/T5ForConditionalGeneration__0_forward_169.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/tx/ctx4uhzyikhxrte7fam6djzy5ok6l2vnuour4uuod3vw32mfd5f6.py
# Source Nodes: [add, l__self___encoder_block_0_layer_0_self_attention_q, l__self___encoder_dropout, l__self___encoder_embed_tokens, mean, mul_1, mul_2, pow_1, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add => add
# l__self___encoder_block_0_layer_0_self_attention_q => convert_element_type, view_1
# l__self___encoder_dropout => gt, mul_1, mul_2
# l__self___encoder_embed_tokens => embedding
# mean => mean
# mul_1 => mul_3
# mul_2 => mul_4
# pow_1 => pow_1
# rsqrt => rsqrt
triton_per_fused__to_copy_add_embedding_mean_mul_native_dropout_pow_rsqrt_view_0 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_native_dropout_pow_rsqrt_view_0', '''
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
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_native_dropout_pow_rsqrt_view_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_native_dropout_pow_rsqrt_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tl.where(tmp5 < 0, tmp5 + 32128, tmp5)
    # tl.device_assert((0 <= tmp6) & (tmp6 < 32128), "index out of bounds: 0 <= tmp6 < 32128")
    tmp7 = tl.load(in_ptr2 + (r1 + (512*tmp6)), rmask, other=0.0)
    tmp8 = tmp4.to(tl.float32)
    tmp9 = tmp8 * tmp7
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp23 = tmp11 * tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp4, rmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp7, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp25, rmask)
''')
