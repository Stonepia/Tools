

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tr/ctrhxss2k7jbooxekyvu5o7vc2yd2py6vamjq6emeum6lviabphh.py
# Source Nodes: [add, l__self___embed_tokens, mean, mul_1, mul_2, pow_1, rsqrt, to_1, to_2], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add
# l__self___embed_tokens => embedding
# mean => mean
# mul_1 => mul_1
# mul_2 => mul_2
# pow_1 => pow_1
# rsqrt => rsqrt
# to_1 => convert_element_type_1
# to_2 => convert_element_type_2
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 32128), "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert((0 <= tmp9) & (tmp9 < 32128), "index out of bounds: 0 <= tmp9 < 32128")
        tmp10 = tl.load(in_ptr1 + (r1 + (512*tmp9)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 512.0
        tmp13 = tmp6 / tmp12
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp8 * tmp18
        tl.store(out_ptr1 + (r1 + (512*x0)), tmp19, rmask)
''')
