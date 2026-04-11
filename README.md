# FLUX Optimizer — Peephole Bytecode Optimizer

Reduce FLUX bytecode size and improve performance.

## Optimization Passes
- **HALT Truncate**: Remove dead code after first HALT
- **NOP Eliminate**: Remove no-op instructions
- **Constant Fold**: MOVI + ADDI → MOVI (combined immediate)
- **Strength Reduce**: MUL by 2 → ADD (register-to-register)
- **Redundant MOV**: Remove MOV R0, R0 self-assignments
- **Dead Code**: Remove unused register stores
- **Peephole**: General pattern matching

## Usage

```python
from optimizer import PeepholeOptimizer, OptPass

opt = PeepholeOptimizer([0x01, 0x18, 0, 10, 0x19, 0, 20, 0x00, 0x18, 1, 99])
result = opt.optimize()
print(f"Saved {result.savings} bytes ({result.original_bytes} → {result.optimized_bytes})")
for rule in result.rules_applied:
    print(f"  {rule}")
```

10 tests passing.
