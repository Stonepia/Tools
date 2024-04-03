

# Original file: ./MegatronBertForCausalLM__0_backward_279.1/MegatronBertForCausalLM__0_backward_279.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/62/c624i6l7di3z5t4xqw62xkpn5y72lu2jyepmca5zqfjun63lc5aj.py
# Source Nodes: [cross_entropy], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.nll_loss_forward, aten.sum]
# cross_entropy => full_default_3
triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_33 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_33', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*i64', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_33', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_nll_loss_forward_sum_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr0 + (524288 + x2), None)
    tmp13 = tl.load(in_ptr1 + (512 + x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (524288 + x2), None)
    tmp18 = tl.load(in_ptr3 + (524288 + x2), None)
    tmp23 = tl.load(in_ptr0 + (1048576 + x2), None)
    tmp24 = tl.load(in_ptr1 + (1024 + x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (1048576 + x2), None)
    tmp29 = tl.load(in_ptr3 + (1048576 + x2), None)
    tmp34 = tl.load(in_ptr0 + (1572864 + x2), None)
    tmp35 = tl.load(in_ptr1 + (1536 + x1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (1572864 + x2), None)
    tmp40 = tl.load(in_ptr3 + (1572864 + x2), None)
    tmp45 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1024.0
    tmp3 = tmp1 / tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp14 = tmp13 / tmp2
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 + tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp9
    tmp21 = tmp17 * tmp20
    tmp22 = tmp11 + tmp21
    tmp25 = tmp24 / tmp2
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp9
    tmp32 = tmp28 * tmp31
    tmp33 = tmp22 + tmp32
    tmp36 = tmp35 / tmp2
    tmp38 = tmp36 * tmp37
    tmp39 = tmp34 + tmp38
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp9
    tmp43 = tmp39 * tmp42
    tmp44 = tmp33 + tmp43
    tmp46 = tl.where(tmp45 < 0, tmp45 + 512, tmp45)
    tmp47 = tl.full([1], -1, tl.int64)
    tmp48 = tmp45 == tmp47
    tmp49 = 0.0
    tmp50 = tl.where(tmp48, tmp49, tmp44)
    tl.atomic_add(out_ptr1 + (x0 + (1024*tmp46)), tmp50, None)
''')
