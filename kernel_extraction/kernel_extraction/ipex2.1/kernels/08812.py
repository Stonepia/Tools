

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/3z/c3zzjrzx4526jmxcnli6ohstk566nscpdf3uw47npddlrieob3nm.py
# Source Nodes: [add, layer_norm], Original ATen: [aten.add, aten.native_layer_norm]
# add => add
# layer_norm => add_1, add_2, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_add_native_layer_norm_1 = async_compile.triton('triton_red_fused_add_native_layer_norm_1', '''
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
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196) % 16
    x2 = (xindex // 3136)
    x4 = xindex % 3136
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*(x0 % 14)) + (1792*(x1 % 4)) + (7168*(x0 // 14)) + (100352*(x1 // 4)) + (401408*x2)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp8 = tl.load(in_ptr0 + (r3 + (128*(x0 % 14)) + (1792*(x1 % 4)) + (7168*(x0 // 14)) + (100352*(x1 // 4)) + (401408*x2)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp5
        tmp13 = 128.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-06
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (128*x5)), tmp25, rmask)
''')
