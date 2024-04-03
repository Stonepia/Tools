

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/ob/cobnw4mqos7gqvyojczstwlf52mpo5gluj7hel7bhy7gh2qs3p7y.py
# Source Nodes: [add_4, add_5, l__self___block_0_layer_0_dropout, l__self___embed_tokens, mean_1, mul_5, mul_6, pow_2, rsqrt_1], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_4 => add_6
# add_5 => add_7
# l__self___block_0_layer_0_dropout => clone_3
# l__self___embed_tokens => embedding
# mean_1 => mean_1
# mul_5 => mul_5
# mul_6 => mul_6
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_4', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
    tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, other=0.0)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 512.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp4 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
''')
