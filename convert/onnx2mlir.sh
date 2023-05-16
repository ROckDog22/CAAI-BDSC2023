model_transform.py \
--model_name uie \
--model_def inference.onnx \
--input_shapes "[[1,35], [1,35], [1,35]]" \
--pixel_format rgb \
--model_format nlp \
--mlir transformed.mlir
