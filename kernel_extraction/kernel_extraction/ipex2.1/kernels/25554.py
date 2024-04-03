

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/mp/cmpzmg7khytolsiqpkwn65n4gu7onbqmycxqqngwuk3bglznfu34.py
# Source Nodes: [add_12, add_17, contiguous_8, l__self___image_encoder_blocks_2_mlp_lin1, l__self___image_encoder_blocks_2_norm2], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.native_layer_norm]
# add_12 => add_22
# add_17 => add_29
# contiguous_8 => clone_23
# l__self___image_encoder_blocks_2_mlp_lin1 => convert_element_type_50
# l__self___image_encoder_blocks_2_norm2 => add_30, add_31, clone_24, mul_31, mul_32, rsqrt_5, sub_15, var_mean_5
triton_red_fused__to_copy_add_clone_native_layer_norm_17 = async_compile.triton('triton_red_fused__to_copy_add_clone_native_layer_norm_17', '''
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_clone_native_layer_norm_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_clone_native_layer_norm_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1280*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (1280*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r2 + (1280*(x0 % 14)) + (17920*(x1 % 14)) + (250880*(x0 // 14)) + (1254400*(x1 // 14))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight,
        )
        tmp8_mean = tl.where(rmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r2 + (1280*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r2 + (1280*(x0 % 14)) + (17920*(x1 % 14)) + (250880*(x0 // 14)) + (1254400*(x1 // 14))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp11 + tmp13
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp18 = tmp17 - tmp8
        tmp19 = 1280.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (1280*x3)), tmp29, rmask)
''')
