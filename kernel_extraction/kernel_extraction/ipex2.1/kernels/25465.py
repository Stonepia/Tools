

# Original file: ./sam___60.0/sam___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/bf/cbfvsptsr33meq6dj6yll35yqvhkdz3ytdigxhboubeawcj32d6i.py
# Source Nodes: [add_204, add_208, l__mod___mask_decoder_transformer_layers_0_norm4], Original ATen: [aten.add, aten.native_layer_norm]
# add_204 => add_373
# add_208 => add_381
# l__mod___mask_decoder_transformer_layers_0_norm4 => add_374, add_375, clone_250, convert_element_type_334, convert_element_type_335, mul_394, mul_395, rsqrt_67, sub_174, var_mean_67
triton_red_fused_add_native_layer_norm_51 = async_compile.triton('triton_red_fused_add_native_layer_norm_51', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_51', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r1 + (256*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp5
        tmp13 = 256.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-05
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r1 + (256*x0)), tmp25, rmask)
        tl.store(out_ptr3 + (r1 + (256*x0)), tmp27, rmask)
''')
