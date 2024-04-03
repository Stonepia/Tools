

# Original file: ./MegatronBertForCausalLM__0_backward_207.1/MegatronBertForCausalLM__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/hk/chkfh5jwkhsf3ocv67qwtoa4kev2kclo5jbr4kwsfodveiujhbdj.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.sum]

triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_25 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*i64', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_sum_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (x2), None)
    tmp13 = tl.load(in_ptr0 + (524288 + x2), None).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (512 + x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (524288 + x2), None)
    tmp20 = tl.load(in_ptr3 + (524288 + x2), None)
    tmp25 = tl.load(in_ptr0 + (1048576 + x2), None).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (1024 + x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (1048576 + x2), None)
    tmp32 = tl.load(in_ptr3 + (1048576 + x2), None)
    tmp37 = tl.load(in_ptr0 + (1572864 + x2), None).to(tl.float32)
    tmp38 = tl.load(in_ptr1 + (1536 + x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr2 + (1572864 + x2), None)
    tmp44 = tl.load(in_ptr3 + (1572864 + x2), None)
    tmp49 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = 1024.0
    tmp3 = tmp1 / tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp7 * tmp11
    tmp15 = tmp14 / tmp2
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp13 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp10
    tmp23 = tmp19 * tmp22
    tmp24 = tmp12 + tmp23
    tmp27 = tmp26 / tmp2
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp25 + tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp10
    tmp35 = tmp31 * tmp34
    tmp36 = tmp24 + tmp35
    tmp39 = tmp38 / tmp2
    tmp41 = tmp39 * tmp40
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp37 + tmp42
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp45 * tmp10
    tmp47 = tmp43 * tmp46
    tmp48 = tmp36 + tmp47
    tmp50 = tl.where(tmp49 < 0, tmp49 + 512, tmp49)
    tmp51 = tl.full([1], -1, tl.int64)
    tmp52 = tmp49 == tmp51
    tmp53 = tmp48.to(tl.float32)
    tmp54 = 0.0
    tmp55 = tl.where(tmp52, tmp54, tmp53)
    tl.atomic_add(out_ptr1 + (x0 + (1024*tmp50)), tmp55, None)
''')
