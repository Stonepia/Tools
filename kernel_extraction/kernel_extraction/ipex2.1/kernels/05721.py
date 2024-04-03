

# Original file: ./botnet26t_256___60.0/botnet26t_256___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/4e/c4ed3542wswmxuecaucdztsftrwb5eh6r7grfdn2axocbzs6iqop.py
# Source Nodes: [add_12, mul_2, softmax_2], Original ATen: [aten._softmax, aten.add, aten.mul]
# add_12 => add_70
# mul_2 => mul_89
# softmax_2 => amax_2, convert_element_type_91, convert_element_type_92, div_2, exp_2, sub_31, sum_3
triton_per_fused__softmax_add_mul_20 = async_compile.triton('triton_per_fused__softmax_add_mul_20', '''
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
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_mul_20(in_ptr0, in_ptr1, in_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = 0.08838834764831845
    tmp2 = tmp0 * tmp1
    tmp3 = 7 + (15*(x0 // 8)) + (r2 // 8)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (7 + (15*(x0 // 8)) + (r2 // 8)) % 16
    tmp7 = tl.full([1, 1], 15, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((15*((7 + (15*(x0 // 8)) + (r2 // 8)) // 16)) + (120*(x0 % 8)) + (960*((((8*x1) + (x0 % 8)) // 8) % 512)) + ((7 + (15*(x0 // 8)) + (r2 // 8)) % 16)), rmask & tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp9, tmp10, 0.0)
    tmp12 = tl.where(tmp5, tmp11, 0.0)
    tmp13 = 7 + (15*(x0 % 8)) + (r2 % 8)
    tmp14 = tmp13 < tmp4
    tmp15 = (7 + (15*(x0 % 8)) + (r2 % 8)) % 16
    tmp16 = tmp15 < tmp7
    tmp17 = tmp16 & tmp14
    tmp18 = tl.load(in_ptr2 + ((15*(((7 + (15*(x0 % 8)) + (r2 % 8)) // 16) % 8)) + (120*(x0 // 8)) + (960*x1) + ((7 + (15*(x0 % 8)) + (r2 % 8)) % 16)), rmask & tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tl.where(tmp14, tmp19, 0.0)
    tmp21 = tmp12 + tmp20
    tmp22 = tmp2 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask, tmp24, float("-inf"))
    tmp27 = triton_helpers.max2(tmp26, 1)[:, None]
    tmp28 = tmp23 - tmp27
    tmp29 = tl.exp(tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None]
    tmp34 = tmp29 / tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr3 + (r2 + (64*x3)), tmp35, rmask)
''')
