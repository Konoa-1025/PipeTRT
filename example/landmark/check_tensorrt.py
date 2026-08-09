# scripts/check_tensorrt.py

import tensorrt as trt

def main():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    print(f"TensorRT version: {trt.__version__}")
    print(f"DLA cores: {builder.num_DLA_cores}")
    print("TensorRT Builder initialization: OK")


if __name__ == "__main__":
    main()