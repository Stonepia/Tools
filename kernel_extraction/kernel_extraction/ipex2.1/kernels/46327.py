

# Original file: ./PegasusForConditionalGeneration__54_backward_367.41/PegasusForConditionalGeneration__54_backward_367.41_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/5c/c5cgqkgmspaze2qxl23agqggnvbdsfkdflnjijxyfitrd5pylmrq.py
# Source Nodes: [add, l__self___final_layer_norm, l__self___self_attn_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# add => add_2
# l__self___final_layer_norm => mul_5, sub_2
# l__self___self_attn_layer_norm => mul, sub
triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr6 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr7 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp2 + tmp4
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp23 = tmp2 - tmp22
        tmp25 = tmp23 * tmp24
        tmp26 = tmp21 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
        tmp30 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
        tmp33 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, None)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp31, None)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp34, None)
''')
