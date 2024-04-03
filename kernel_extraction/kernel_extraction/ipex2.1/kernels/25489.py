

# Original file: ./sam___60.0/sam___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/xy/cxydpk2m774upugafovqd2eyq52nd5zzmkpzhjmdefm2wnfsytbe.py
# Source Nodes: [add, add_11, add_5, add_6, contiguous_2, contiguous_5, l__mod___image_encoder_blocks_1_norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add
# add_11 => add_18
# add_5 => add_7
# add_6 => add_11
# contiguous_2 => clone_7
# contiguous_5 => clone_15
# l__mod___image_encoder_blocks_1_norm2 => add_19, add_20, clone_16, convert_element_type_16, convert_element_type_17, mul_19, mul_20, rsqrt_3, sub_10, var_mean_3
triton_red_fused_add_clone_native_layer_norm_13 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_13', '''
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
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (1280*(x0 % 14)) + (17920*(x1 % 14)) + (250880*(x0 // 14)) + (1254400*(x1 // 14))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r2 + (1280*(x0 % 14)) + (17920*(x1 % 14)) + (250880*(x0 // 14)) + (1254400*(x1 // 14))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight,
        )
        tmp11_mean = tl.where(rmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask, tmp11_weight_next, tmp11_weight)
        tl.store(out_ptr0 + (r2 + (1280*x3)), tmp8, rmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(out_ptr0 + (r2 + (1280*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight,
        )
        tmp17_mean = tl.where(rmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask, tmp17_weight_next, tmp17_weight)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(out_ptr0 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21 - tmp11
        tmp23 = 1280.0
        tmp24 = tmp18 / tmp23
        tmp25 = 1e-06
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (1280*x3)), tmp35, rmask)
''')
