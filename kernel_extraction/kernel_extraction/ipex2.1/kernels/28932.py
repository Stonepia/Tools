

# Original file: ./mobilevit_s___60.0/mobilevit_s___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/ya/cya3yuskjfineszrdtuakb3ggc75ze4ec333m4rbpiz4wcy7w2ai.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => add_100, add_101, convert_element_type_183, convert_element_type_184, mul_155, mul_156, rsqrt_14, sub_49, var_mean_14
triton_red_fused_native_layer_norm_48 = async_compile.triton('triton_red_fused_native_layer_norm_48', '''
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
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_48', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_48(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((240*(((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + ((x0 % 4) % 2)) % 8)) + (1920*((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 8) % 8)) + (15360*((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 15360) % 64)) + ((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 64) % 240)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight,
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + ((240*(((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + ((x0 % 4) % 2)) % 8)) + (1920*((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 8) % 8)) + (15360*((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 15360) % 64)) + ((((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 61440)) + ((x0 % 4) % 2)) // 64) % 240)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 - tmp3
        tmp9 = 240.0
        tmp10 = tmp4 / tmp9
        tmp11 = 1e-05
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.rsqrt(tmp12)
        tmp14 = tmp8 * tmp13
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (240*x1) + (3840*x0)), tmp21, rmask)
''')
