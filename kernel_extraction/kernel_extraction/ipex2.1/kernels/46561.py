

# Original file: ./moondream___60.0/moondream___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/iq/ciq27m7leburarxea5udfbut54lcikuxolpg4vtunrstfgpc7g52.py
# Source Nodes: [l__mod___model_embed_tokens, l__mod___model_layers_0_input_layernorm], Original ATen: [aten.embedding, aten.native_layer_norm]
# l__mod___model_embed_tokens => embedding
# l__mod___model_layers_0_input_layernorm => add_1, add_2, convert_element_type_1, convert_element_type_2, mul, mul_1, rsqrt, sub, var_mean
triton_red_fused_embedding_native_layer_norm_0 = async_compile.triton('triton_red_fused_embedding_native_layer_norm_0', '''
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_native_layer_norm_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 51200, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp1 < 51200")
        tmp2 = tl.load(in_ptr1 + (r1 + (2048*tmp1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight,
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
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
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.where(tmp0 < 0, tmp0 + 51200, tmp0)
        # tl.device_assert(((0 <= tmp8) & (tmp8 < 51200)) | ~xmask, "index out of bounds: 0 <= tmp8 < 51200")
        tmp9 = tl.load(in_ptr1 + (r1 + (2048*tmp8)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp5
        tmp12 = 2048.0
        tmp13 = tmp6 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp24, rmask & xmask)
''')
