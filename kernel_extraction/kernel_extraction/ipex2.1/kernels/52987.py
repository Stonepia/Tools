

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/rc/crcgrh4ufvejxdugmv4scnlz4derf6ygax7dfkre7id4kavpcgbm.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.sum]

triton_red_fused__to_copy_add_sum_16 = async_compile.triton('triton_red_fused__to_copy_add_sum_16', '''
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
    size_hints=[16384, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_sum_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_sum_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp42 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp46 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr4 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr5 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr6 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr7 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr8 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp36 = tl.load(in_ptr9 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_ptr10 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp44 = tl.load(in_ptr11 + (x0 + (16384*r1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(rmask, tmp43, _tmp42)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
        tmp47 = _tmp46 + tmp45
        _tmp46 = tl.where(rmask, tmp47, _tmp46)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp46 = tl.sum(_tmp46, 1)[:, None]
    tmp48 = tmp2.to(tl.float32)
    tmp49 = tmp6.to(tl.float32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp10.to(tl.float32)
    tmp52 = tmp50 + tmp51
    tmp53 = tmp14.to(tl.float32)
    tmp54 = tmp52 + tmp53
    tmp55 = tmp18.to(tl.float32)
    tmp56 = tmp54 + tmp55
    tmp57 = tmp22.to(tl.float32)
    tmp58 = tmp56 + tmp57
    tmp59 = tmp30.to(tl.float32)
    tmp60 = tmp58 + tmp59
    tmp61 = tmp38.to(tl.float32)
    tmp62 = tmp60 + tmp61
    tmp63 = tmp46.to(tl.float32)
    tmp64 = tmp62 + tmp63
    tmp65 = tmp26.to(tl.float32)
    tmp66 = tmp64 + tmp65
    tmp67 = tmp34.to(tl.float32)
    tmp68 = tmp66 + tmp67
    tmp69 = tmp42.to(tl.float32)
    tmp70 = tmp68 + tmp69
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp70, None)
''')
