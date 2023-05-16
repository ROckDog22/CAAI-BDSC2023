model_deploy.py \
--mlir transformed.mlir \
--quantize F32 \
--chip bm1684x \
--tolerance 0.80,0.80 \
--model uie-nano.bmodel;
