

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ql/cqlbvidue3ju3cnepzfi3o2segk32qgl5g5bfoaxlpaz4ghjwetd.py
# Source Nodes: [add_4, add_5, l__mod___model_encoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens, mean_1, mul_5, mul_6, pow_2, rsqrt_1, to_5, to_6], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_6
# add_5 => add_7
# l__mod___model_encoder_block_0_layer_0_dropout => clone_3
# l__mod___model_encoder_embed_tokens => embedding
# mean_1 => mean_1
# mul_5 => mul_5
# mul_6 => mul_6
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
# to_5 => convert_element_type_8
# to_6 => convert_element_type_9
triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_6', '''
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
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert(((0 <= tmp1) & (tmp1 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*tmp1)), rmask & xmask, other=0.0).to(tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 1024.0
    tmp13 = tmp10 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp5 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp11 * tmp18
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp19, rmask & xmask)
''')
