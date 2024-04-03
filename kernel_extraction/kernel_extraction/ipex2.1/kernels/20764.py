

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/uv/cuvqg47kll4dajlisksqcewqkecpr25tdadp2nogtsn6v4bl33s5.py
# Source Nodes: [cross_entropy], Original ATen: [aten._to_copy, aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_per_fused__to_copy_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_29 = async_compile.triton('triton_per_fused__to_copy_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_29', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i64', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_29', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_embedding_dense_backward_native_layer_norm_backward_nll_loss_forward_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp9 = tmp3 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp3 * tmp15
    tmp17 = tmp16 - tmp7
    tmp18 = tmp8 * tmp13
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp22 = tl.where(tmp21 < 0, tmp21 + 2, tmp21)
    tmp23 = tl.full([1, 1], -1, tl.int64)
    tmp24 = tmp21 == tmp23
    tmp25 = 0.0
    tmp26 = tl.where(tmp24, tmp25, tmp20)
    tmp28 = tl.where(tmp27 < 0, tmp27 + 30000, tmp27)
    tmp29 = tl.full([1, 1], 0, tl.int64)
    tmp30 = tmp27 == tmp29
    tmp31 = tl.where(tmp30, tmp25, tmp20)
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp20, rmask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (128*tmp22), [XBLOCK, RBLOCK])), tmp26, rmask)
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (128*tmp28), [XBLOCK, RBLOCK])), tmp31, rmask)
''')
