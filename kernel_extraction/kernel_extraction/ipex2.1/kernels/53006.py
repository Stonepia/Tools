

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/k3/ck3vw3ijzvx2fxrihqhkgvfgampelevqrfosijqi7gdqqurcj5cp.py
# Source Nodes: [l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_11], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_11 => convert_element_type_73
triton_red_fused_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_5', '''
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
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tmp17
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp18 = 4096.0
        tmp19 = tmp12 / tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = tmp24 * tmp18
        tmp26 = tmp25 - tmp6
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp10
        tmp30 = tmp29 * tmp12
        tmp31 = tmp30 * tmp16
        tmp32 = tmp26 - tmp31
        tmp33 = tmp19 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp34, None)
''')