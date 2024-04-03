

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/dx/cdxqofkyid7szredeu6qdhtnxxqylg62cfgrv6fc3wdad6revd3j.py
# Source Nodes: [l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_11], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_11 => convert_element_type_229
triton_red_fused_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_11', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tmp3 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tmp16
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp17 = 4096.0
        tmp18 = tmp11 / tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22 * tmp17
        tmp24 = tmp23 - tmp5
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp9
        tmp28 = tmp27 * tmp11
        tmp29 = tmp28 * tmp15
        tmp30 = tmp24 - tmp29
        tmp31 = tmp18 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp32, None)
''')
