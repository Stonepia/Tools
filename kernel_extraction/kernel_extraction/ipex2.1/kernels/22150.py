

# Original file: ./gmlp_s16_224___60.0/gmlp_s16_224___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/57/c573kal6iqlqyttmihu4pgwxnuvpkxabp32qrz7zlan3oud3pc3s.py
# Source Nodes: [add_28, add_29, getattr_l__mod___blocks___28___mlp_channels_drop2, getattr_l__mod___blocks___29___mlp_channels_drop2, l__mod___norm, mean], Original ATen: [aten.add, aten.clone, aten.mean, aten.native_layer_norm]
# add_28 => add_202
# add_29 => add_209
# getattr_l__mod___blocks___28___mlp_channels_drop2 => clone_144
# getattr_l__mod___blocks___29___mlp_channels_drop2 => clone_149
# l__mod___norm => add_210, add_211, convert_element_type_180, convert_element_type_181, mul_240, mul_241, rsqrt_60, sub_60, var_mean_60
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_mean_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_mean_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2 + (196*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp9 = 256.0
        tmp10 = tmp8 / tmp9
        tmp11 = 1e-06
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.rsqrt(tmp12)
        tmp14 = tmp7 * tmp13
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = 196.0
    tmp27 = tmp24 / tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp28, None)
''')
