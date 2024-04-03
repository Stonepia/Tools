

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/w2/cw2ynmkhte35gxkblxkayk4ubz6vd2b6vctxfili5qss5z47tqrq.py
# Source Nodes: [add, add_5, dropout_1, dropout_5, l__self___encoder_layers_1_feed_forward_layer_norm], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add => add_3
# add_5 => add_13
# dropout_1 => clone_4
# dropout_5 => clone_9
# l__self___encoder_layers_1_feed_forward_layer_norm => add_14, add_15, convert_element_type_12, convert_element_type_13, mul_10, mul_9, rsqrt_4, sub_7, var_mean_3
triton_red_fused_add_clone_native_layer_norm_28 = async_compile.triton('triton_red_fused_add_clone_native_layer_norm_28', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight,
        )
        tmp7_mean = tl.where(rmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask, tmp7_weight_next, tmp7_weight)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr0 + (256 + r1 + (512*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp7
        tmp17 = 256.0
        tmp18 = tmp8 / tmp17
        tmp19 = 1e-12
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (256*x0)), tmp29, rmask)
''')
