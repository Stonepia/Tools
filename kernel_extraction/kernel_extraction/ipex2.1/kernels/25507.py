

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/3m/c3my7zn3hhtyqhkxwosrtaaymben325gvva5fnpumkananrgcv7c.py
# Source Nodes: [add_193, add_194, mean, mean_1, mul_160, pow_1, sqrt, sub_65, truediv_1], Original ATen: [aten.add, aten.div, aten.mean, aten.mul, aten.pow, aten.sqrt, aten.sub]
# add_193 => add_353
# add_194 => add_354
# mean => mean
# mean_1 => mean_1
# mul_160 => mul_384
# pow_1 => pow_1
# sqrt => sqrt
# sub_65 => sub_161
# truediv_1 => div_33
triton_per_fused_add_div_mean_mul_pow_sqrt_sub_32 = async_compile.triton('triton_per_fused_add_div_mean_mul_pow_sqrt_sub_32', '''
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_32', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_sqrt_sub_32(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp15 / tmp6
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sqrt(tmp20)
    tmp22 = tmp9 / tmp21
    tmp23 = tmp16 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp25, rmask)
''')