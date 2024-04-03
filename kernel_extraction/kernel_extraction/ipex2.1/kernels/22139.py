

# Original file: ./gmlp_s16_224___60.0/gmlp_s16_224___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/ky/ckyrofrtsxbboaj7kbmwdgtui5wsxvwvhljvjcq7wyqjahny4xm5.py
# Source Nodes: [add_28, add_29, getattr_l__mod___blocks___28___mlp_channels_drop2, getattr_l__mod___blocks___29___mlp_channels_drop2, l__mod___norm, mean], Original ATen: [aten.add, aten.clone, aten.mean, aten.native_layer_norm]
# add_28 => add_202
# add_29 => add_209
# getattr_l__mod___blocks___28___mlp_channels_drop2 => clone_144
# getattr_l__mod___blocks___29___mlp_channels_drop2 => clone_149
# l__mod___norm => add_210, add_211, mul_240, mul_241, rsqrt_60, sub_60, var_mean_60
# mean => mean
triton_red_fused_add_clone_mean_native_layer_norm_10 = async_compile.triton('triton_red_fused_add_clone_mean_native_layer_norm_10', '''
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
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_mean_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_mean_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr4 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 - tmp5
        tmp8 = 256.0
        tmp9 = tmp7 / tmp8
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = 196.0
    tmp22 = tmp19 / tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, None)
''')
