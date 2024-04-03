

# Original file: ./tnt_s_patch16_224___60.0/tnt_s_patch16_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ak/cakun6so65ikcbc7xfefk3jqskljbzv6qix5ashtcieh5k7rs6sc.py
# Source Nodes: [l__mod___norm1_proj], Original ATen: [aten.native_layer_norm]
# l__mod___norm1_proj => add_3, add_4, convert_element_type_2, convert_element_type_3, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_native_layer_norm_1 = async_compile.triton('triton_red_fused_native_layer_norm_1', '''
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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((24*((r2 // 24) % 4)) + (24*(tl.where(((4*(x0 % 14)) + ((r2 // 24) % 4)) >= 0, 0, 56))) + (96*(x0 % 14)) + (1344*((((4*(r2 // 96)) + ((r2 // 24) % 4)) // 4) % 4)) + (1344*(tl.where(((4*(x0 // 14)) + ((((4*(r2 // 96)) + ((r2 // 24) % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*(x0 // 14)) + (75264*x1) + ((((4*(r2 // 96)) + (16*(r2 % 24)) + ((r2 // 24) % 4)) // 16) % 24)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + ((16*(r2 % 24)) + (r2 // 24)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight,
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + ((24*((r2 // 24) % 4)) + (24*(tl.where(((4*(x0 % 14)) + ((r2 // 24) % 4)) >= 0, 0, 56))) + (96*(x0 % 14)) + (1344*((((4*(r2 // 96)) + ((r2 // 24) % 4)) // 4) % 4)) + (1344*(tl.where(((4*(x0 // 14)) + ((((4*(r2 // 96)) + ((r2 // 24) % 4)) // 4) % 4)) >= 0, 0, 56))) + (5376*(x0 // 14)) + (75264*x1) + ((((4*(r2 // 96)) + (16*(r2 % 24)) + ((r2 // 24) % 4)) // 16) % 24)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + ((16*(r2 % 24)) + (r2 // 24)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp7
        tmp17 = 384.0
        tmp18 = tmp8 / tmp17
        tmp19 = 1e-05
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp29, rmask & xmask)
''')
