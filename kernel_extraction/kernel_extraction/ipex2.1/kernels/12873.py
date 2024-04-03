

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/o7/co7ufnghek6hf6alcncpbv2ewpit5yj7e6ry2epeihsh3ufqvpbo.py
# Source Nodes: [add, add_5, add_7, dropout_1, dropout_5, dropout_9, l__self___encoder_layers_2_feed_forward_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_3
# add_5 => add_13
# add_7 => add_20
# dropout_1 => clone_4
# dropout_5 => clone_9
# dropout_9 => clone_15
# l__self___encoder_layers_2_feed_forward_layer_norm => add_21, add_22, mul_13, mul_14, rsqrt_6, sub_11, var_mean_5
triton_red_fused_add_clone_native_layer_norm_30 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_30', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
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
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 - tmp8
        tmp19 = 256.0
        tmp20 = tmp9 / tmp19
        tmp21 = 1e-12
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp26 = tmp24 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr2 + (r1 + (256*x0)), tmp28, rmask)
''')
