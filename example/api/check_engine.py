from pipetrt.engines.manager import EngineManager


manager = EngineManager(
    model="full",
    precision="fp16"
)

engines = manager.ensure_engines()

print("Palm Engine:")
print(engines["palm"])

print()

print("Landmark Engine:")
print(engines["landmark"])