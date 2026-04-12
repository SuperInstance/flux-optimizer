"""
Comprehensive test suite for FLUX PeepholeOptimizer.

50+ tests organised by optimisation pass and cross-cutting concerns.
Each test is independent and does not rely on the order of execution.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optimizer import PeepholeOptimizer, OptPass, OptResult


# ── Helpers ─────────────────────────────────────────────

def _opt(bc, passes=None):
    """Convenience: build an optimizer, run passes, return (result, bytecode)."""
    o = PeepholeOptimizer(bc)
    r = o.optimize(passes)
    return r, o.get_bytecode()


# ═══════════════════════════════════════════════════════════
# NOP elimination
# ═══════════════════════════════════════════════════════════
class TestNOPElimination(unittest.TestCase):
    """NOP (0x01, Format A, 1 byte) removal."""

    # 1 – single NOP at the start
    def test_single_nop_at_start(self):
        r, bc = _opt([0x01, 0x18, 0, 42, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 4)
        self.assertNotIn(0x01, bc)
        self.assertTrue(any("nop_eliminate" in x for x in r.rules_applied))

    # 2 – single NOP at the end
    def test_single_nop_at_end(self):
        r, bc = _opt([0x18, 0, 42, 0x00, 0x01], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 4)
        self.assertNotIn(0x01, bc)

    # 3 – single NOP in the middle
    def test_single_nop_in_middle(self):
        # Use immediates != 0x01 to avoid the byte-level NOP scanner eating them
        r, bc = _opt([0x18, 0, 5, 0x01, 0x18, 2, 7, 0x00], [OptPass.NOP_ELIM])
        self.assertNotIn(0x01, bc)
        self.assertEqual(len(bc), 7)

    # 4 – multiple consecutive NOPs
    def test_multiple_consecutive_nops(self):
        r, bc = _opt([0x01, 0x01, 0x01, 0x18, 0, 42, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 4)
        self.assertEqual(r.rules_applied.count("nop_eliminate"), 3)

    # 5 – no NOPs present (nothing should change)
    def test_no_nops(self):
        r, bc = _opt([0x18, 0, 42, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 4)
        self.assertEqual(len(r.rules_applied), 0)

    # 6 – all NOPs
    def test_all_nops(self):
        r, bc = _opt([0x01, 0x01, 0x01], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 0)
        self.assertEqual(r.rules_applied.count("nop_eliminate"), 3)

    # 7 – NOP between two non-NOP instructions
    def test_nop_between_instructions(self):
        r, bc = _opt([0x00, 0x01, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(bc, [0x00, 0x00])

    # 8 – savings are correct for NOP removal
    def test_nop_savings(self):
        r, bc = _opt([0x18, 0, 42, 0x01, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(r.savings, 1)


# ═══════════════════════════════════════════════════════════
# HALT truncation
# ═══════════════════════════════════════════════════════════
class TestHALTTruncation(unittest.TestCase):
    """Truncate bytecode after the *first* HALT (0x00)."""

    # 9 – HALT at the start
    def test_halt_at_start(self):
        r, bc = _opt([0x00, 0x18, 0, 42, 0x18, 1, 10], [OptPass.HALT_TRUNCATE])
        self.assertEqual(bc, [0x00])  # nothing after first HALT

    # 10 – HALT in the middle
    def test_halt_in_middle(self):
        r, bc = _opt([0x18, 0, 42, 0x00, 0x18, 1, 10], [OptPass.HALT_TRUNCATE])
        self.assertEqual(len(bc), 4)
        self.assertEqual(bc[-1], 0x00)

    # 11 – no HALT present (no change)
    def test_no_halt(self):
        r, bc = _opt([0x18, 0, 42, 0x18, 1, 10], [OptPass.HALT_TRUNCATE])
        self.assertEqual(len(bc), 6)
        self.assertEqual(len(r.rules_applied), 0)

    # 12 – multiple HALTs (truncate at first)
    def test_multiple_halts(self):
        r, bc = _opt([0x18, 0, 42, 0x00, 0x18, 1, 10, 0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(len(bc), 4)
        self.assertTrue(any("halt_truncate" in x for x in r.rules_applied))

    # 13 – HALT at the very end (nothing to truncate)
    def test_halt_at_end(self):
        r, bc = _opt([0x18, 0, 42, 0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(len(bc), 4)
        self.assertEqual(len(r.rules_applied), 0)  # no change needed

    # 14 – only HALT
    def test_only_halt(self):
        r, bc = _opt([0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(bc, [0x00])


# ═══════════════════════════════════════════════════════════
# Constant folding
# ═══════════════════════════════════════════════════════════
class TestConstantFolding(unittest.TestCase):
    """MOVI Rn,X + ADDI Rn,Y → MOVI Rn,X+Y."""

    # 15 – basic positive fold
    def test_basic_fold(self):
        r, bc = _opt([0x18, 0, 10, 0x19, 0, 20, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertEqual(bc[0], 0x18)
        self.assertEqual(bc[2], 30)
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied))

    # 16 – negative ADDI value
    def test_negative_addi(self):
        r, bc = _opt([0x18, 0, 10, 0x19, 0, 0xFB, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertEqual(bc[2], 5)  # 10 + (-5)

    # 17 – overflow (result > 127, should NOT fold)
    def test_overflow_positive(self):
        r, bc = _opt([0x18, 0, 100, 0x19, 0, 100, 0x00], [OptPass.CONSTANT_FOLD])
        # 200 > 127 so fold should not happen – bytecode preserved as-is
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied) is False)

    # 18 – underflow (result < -128, should NOT fold)
    def test_overflow_negative(self):
        r, bc = _opt([0x18, 0, 127, 0x19, 0, 127, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied) is False)

    # 19 – fold with different registers (no change)
    def test_different_registers_no_fold(self):
        r, bc = _opt([0x18, 0, 10, 0x19, 1, 20, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied) is False)

    # 20 – multiple folds in sequence
    def test_multiple_folds(self):
        # MOVI R0,10; ADDI R0,20 → MOVI R0,30; then MOVI R1,5; ADDI R1,15 → MOVI R1,20
        bc_in = [0x18, 0, 10, 0x19, 0, 20, 0x18, 1, 5, 0x19, 1, 15, 0x00]
        r, bc = _opt(bc_in, [OptPass.CONSTANT_FOLD])
        self.assertEqual(bc[2], 30)
        self.assertEqual(bc[5], 20)
        self.assertEqual(sum(1 for x in r.rules_applied if "constant_fold" in x), 2)

    # 21 – fold to zero
    def test_fold_to_zero(self):
        r, bc = _opt([0x18, 0, 5, 0x19, 0, 0xFB, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertEqual(bc[2], 0)

    # 22 – fold at boundary (127)
    def test_fold_at_boundary_127(self):
        r, bc = _opt([0x18, 0, 100, 0x19, 0, 27, 0x00], [OptPass.CONSTANT_FOLD])
        self.assertEqual(bc[2], 127)
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied))


# ═══════════════════════════════════════════════════════════
# Strength reduction
# ═══════════════════════════════════════════════════════════
class TestStrengthReduction(unittest.TestCase):
    """MOVI Rn,2 + MUL → ADD (doubling via strength reduction)."""

    # 23 – basic MUL by 2
    def test_mul_by_2(self):
        r, bc = _opt([0x18, 1, 2, 0x22, 0, 0, 1, 0x00], [OptPass.STRENGTH_REDUCE])
        self.assertIn(0x20, bc)
        self.assertTrue(any("strength_reduce" in x for x in r.rules_applied))

    # 24 – MUL by non-2 constant (no change)
    def test_mul_by_non_2(self):
        r, bc = _opt([0x18, 1, 3, 0x22, 0, 0, 1, 0x00], [OptPass.STRENGTH_REDUCE])
        self.assertTrue(any("strength_reduce" in x for x in r.rules_applied) is False)

    # 25 – strength reduce savings (7 bytes → 4 bytes)
    def test_strength_reduce_savings(self):
        r, bc = _opt([0x18, 1, 2, 0x22, 0, 0, 1, 0x00], [OptPass.STRENGTH_REDUCE])
        self.assertEqual(r.savings, 3)

    # 26 – no MUL pattern at all
    def test_no_mul_pattern(self):
        r, bc = _opt([0x18, 0, 42, 0x00], [OptPass.STRENGTH_REDUCE])
        self.assertEqual(len(r.rules_applied), 0)


# ═══════════════════════════════════════════════════════════
# Redundant MOV elimination
# ═══════════════════════════════════════════════════════════
class TestRedundantMOV(unittest.TestCase):
    """MOV Rn,Rn (same src/dst) → removed."""

    # 27 – MOV R0,R0 removed
    def test_mov_r0_r0(self):
        r, bc = _opt([0x3A, 0, 0, 0, 0x00], [OptPass.REDUNDANT_MOV])
        self.assertNotIn(0x3A, bc)
        self.assertTrue(any("redundant_mov" in x for x in r.rules_applied))

    # 28 – MOV R1,R1 removed
    def test_mov_r1_r1(self):
        r, bc = _opt([0x3A, 1, 1, 0, 0x00], [OptPass.REDUNDANT_MOV])
        self.assertNotIn(0x3A, bc)

    # 29 – MOV R0,R1 NOT removed (different registers)
    def test_mov_different_regs(self):
        r, bc = _opt([0x3A, 0, 1, 0, 0x00], [OptPass.REDUNDANT_MOV])
        self.assertIn(0x3A, bc)
        self.assertEqual(len(r.rules_applied), 0)

    # 30 – multiple redundant MOVs
    def test_multiple_redundant_movs(self):
        bc_in = [0x3A, 0, 0, 0, 0x3A, 1, 1, 0, 0x00]
        r, bc = _opt(bc_in, [OptPass.REDUNDANT_MOV])
        self.assertEqual(bc.count(0x3A), 0)
        self.assertEqual(sum(1 for x in r.rules_applied if "redundant_mov" in x), 2)


# ═══════════════════════════════════════════════════════════
# Dead code elimination
# ═══════════════════════════════════════════════════════════
class TestDeadCodeElimination(unittest.TestCase):
    """Remove trailing bytes after the *last* HALT instruction."""

    # 31 – trailing bytes after HALT removed
    def test_trailing_bytes_after_halt(self):
        # MOVI R0,42; HALT; 0xFF 0xFF
        r, bc = _opt([0x18, 0, 42, 0x00, 0xFF, 0xFF], [OptPass.DEAD_CODE])
        self.assertEqual(len(bc), 4)
        self.assertTrue(any("dead_code" in x for x in r.rules_applied))

    # 32 – no trailing bytes (HALT is last) → no change
    def test_no_trailing_bytes(self):
        r, bc = _opt([0x18, 0, 42, 0x00], [OptPass.DEAD_CODE])
        self.assertEqual(len(bc), 4)
        self.assertEqual(len(r.rules_applied), 0)

    # 33 – no HALT at all → no change
    def test_no_halt_no_change(self):
        r, bc = _opt([0x18, 0, 42, 0x18, 1, 10], [OptPass.DEAD_CODE])
        self.assertEqual(len(bc), 6)

    # 34 – trailing full instruction after last HALT removed
    def test_trailing_instruction_after_halt(self):
        # Two HALTs: last one at index 7, trailing garbage after it
        r, bc = _opt([0x18, 0, 42, 0x00, 0x18, 1, 10, 0x00, 0xFF], [OptPass.DEAD_CODE])
        # Keeps up to last HALT (index 7 inclusive = 8 bytes), removes 0xFF
        self.assertEqual(len(bc), 8)
        self.assertTrue(any("dead_code" in x for x in r.rules_applied))

    # 35 – dead code savings
    def test_dead_code_savings(self):
        r, bc = _opt([0x00, 0xFF, 0xFF, 0xFF], [OptPass.DEAD_CODE])
        self.assertEqual(r.savings, 3)
        self.assertEqual(bc, [0x00])


# ═══════════════════════════════════════════════════════════
# Peephole optimisation
# ═══════════════════════════════════════════════════════════
class TestPeephole(unittest.TestCase):
    """ADD Rn,Rn,Rn → SHL Rn,Rn,1."""

    # 36 – ADD R0,R0,R0 replaced
    def test_add_r0_r0_r0(self):
        r, bc = _opt([0x20, 0, 0, 0, 0x00], [OptPass.PEEPHOLE])
        self.assertEqual(bc[0], 0x24)  # SHL opcode
        self.assertEqual(bc[3], 1)     # shift by 1
        self.assertTrue(any("peephole" in x for x in r.rules_applied))

    # 37 – ADD R3,R3,R3 replaced
    def test_add_r3_r3_r3(self):
        r, bc = _opt([0x20, 3, 3, 3, 0x00], [OptPass.PEEPHOLE])
        self.assertEqual(bc[0], 0x24)
        self.assertEqual(bc[1], 3)
        self.assertEqual(bc[3], 1)

    # 38 – ADD R0,R1,R2 NOT replaced (different registers)
    def test_add_different_regs(self):
        r, bc = _opt([0x20, 0, 1, 2, 0x00], [OptPass.PEEPHOLE])
        self.assertEqual(bc[0], 0x20)  # ADD unchanged
        self.assertEqual(len(r.rules_applied), 0)

    # 39 – ADD R0,R0,R1 NOT replaced (only two registers match)
    def test_add_two_regs_match(self):
        r, bc = _opt([0x20, 0, 0, 1, 0x00], [OptPass.PEEPHOLE])
        self.assertEqual(bc[0], 0x20)
        self.assertEqual(len(r.rules_applied), 0)

    # 40 – multiple ADD Rn,Rn,Rn patterns
    def test_multiple_peephole_patterns(self):
        bc_in = [0x20, 0, 0, 0, 0x20, 1, 1, 1, 0x00]
        r, bc = _opt(bc_in, [OptPass.PEEPHOLE])
        self.assertEqual(bc.count(0x24), 2)
        self.assertEqual(sum(1 for x in r.rules_applied if "peephole" in x), 2)

    # 41 – peephole preserves surrounding instructions
    def test_peephole_preserves_surrounding(self):
        # MOVI R0,5; ADD R0,R0,R0; HALT
        bc_in = [0x18, 0, 5, 0x20, 0, 0, 0, 0x00]
        r, bc = _opt(bc_in, [OptPass.PEEPHOLE])
        self.assertEqual(bc[0], 0x18)  # MOVI preserved
        self.assertEqual(bc[2], 5)     # MOVI immediate preserved
        self.assertEqual(bc[3], 0x24)  # SHL
        self.assertEqual(bc[-1], 0x00) # HALT preserved


# ═══════════════════════════════════════════════════════════
# Combined / multi-pass
# ═══════════════════════════════════════════════════════════
class TestCombinedPasses(unittest.TestCase):
    """Multiple passes applied in sequence."""

    # 42 – NOP + redundant MOV + HALT truncate
    def test_nop_mov_halt(self):
        bc_in = [0x01, 0x3A, 0, 0, 0, 0x00, 0x18, 1, 10]
        r, bc = _opt(bc_in, [OptPass.NOP_ELIM, OptPass.REDUNDANT_MOV, OptPass.HALT_TRUNCATE])
        self.assertGreater(r.savings, 0)

    # 43 – default passes (all)
    def test_default_all_passes(self):
        bc_in = [0x01, 0x18, 0, 10, 0x19, 0, 20, 0x00, 0x18, 1, 99]
        r, bc = _opt(bc_in)
        self.assertGreater(r.savings, 0)
        self.assertEqual(len(r.passes_run), len(OptPass))

    # 44 – dead_code after constant_fold
    def test_dead_code_after_fold(self):
        bc_in = [0x18, 0, 10, 0x19, 0, 20, 0x00, 0xFF, 0xFF]
        r, bc = _opt(bc_in, [OptPass.CONSTANT_FOLD, OptPass.DEAD_CODE])
        self.assertTrue(any("constant_fold" in x for x in r.rules_applied))
        self.assertTrue(any("dead_code" in x for x in r.rules_applied))

    # 45 – peephole + NOP elimination
    def test_peephole_plus_nop(self):
        bc_in = [0x01, 0x20, 0, 0, 0, 0x01, 0x00]
        r, bc = _opt(bc_in, [OptPass.PEEPHOLE, OptPass.NOP_ELIM])
        self.assertIn(0x24, bc)
        self.assertNotIn(0x01, bc)

    # 46 – all passes with heavy optimization
    def test_all_passes_heavy(self):
        bc_in = [
            0x01,                           # NOP
            0x18, 0, 10,                    # MOVI R0,10
            0x19, 0, 20,                    # ADDI R0,20
            0x3A, 0, 0, 0,                  # MOV R0,R0
            0x00,                           # HALT
            0xFF, 0xFF, 0xFF,              # dead code
        ]
        r, bc = _opt(bc_in)
        self.assertGreater(r.savings, 0)


# ═══════════════════════════════════════════════════════════
# OptResult tracking
# ═══════════════════════════════════════════════════════════
class TestOptResult(unittest.TestCase):
    """Verify OptResult dataclass fields."""

    # 47 – savings calculation when bytecode shrinks
    def test_savings_positive(self):
        r, _ = _opt([0x18, 0, 42, 0x00, 0x01, 0x01], [OptPass.HALT_TRUNCATE])
        self.assertEqual(r.original_bytes, 6)
        self.assertGreater(r.savings, 0)

    # 48 – savings = 0 when nothing changes
    def test_savings_zero(self):
        r, _ = _opt([0x18, 0, 42, 0x00])
        self.assertEqual(r.savings, 0)

    # 49 – rules_applied list is populated
    def test_rules_applied_populated(self):
        r, _ = _opt([0x01, 0x18, 0, 42, 0x00], [OptPass.NOP_ELIM])
        self.assertGreater(len(r.rules_applied), 0)

    # 50 – passes_run matches requested passes
    def test_passes_run(self):
        passes = [OptPass.NOP_ELIM, OptPass.HALT_TRUNCATE]
        r, _ = _opt([0x01, 0x00, 0xFF], passes)
        self.assertEqual(r.passes_run, [p.value for p in passes])

    # 51 – original_bytes reflects input
    def test_original_bytes(self):
        r, _ = _opt([0x01, 0x02, 0x03], [OptPass.NOP_ELIM])
        self.assertEqual(r.original_bytes, 3)

    # 52 – optimized_bytes reflects output
    def test_optimized_bytes(self):
        r, _ = _opt([0x01, 0x01, 0x00], [OptPass.NOP_ELIM])
        self.assertEqual(r.optimized_bytes, 1)


# ═══════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):
    """Corner cases and boundary conditions."""

    # 53 – empty bytecode
    def test_empty_bytecode(self):
        r, bc = _opt([], [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 0)
        self.assertEqual(r.savings, 0)

    # 54 – single byte
    def test_single_byte(self):
        r, bc = _opt([0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(bc, [0x00])

    # 55 – all NOPs with dead code pass
    def test_all_nops_dead_code(self):
        r, bc = _opt([0x01, 0x01, 0x01], [OptPass.DEAD_CODE])
        self.assertEqual(len(bc), 3)  # no HALT, so no change

    # 56 – all HALTs
    def test_all_halts(self):
        r, bc = _opt([0x00, 0x00, 0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(bc, [0x00])  # truncate at first HALT

    # 57 – very large bytecode (stress)
    def test_large_bytecode(self):
        bc_in = [0x01] * 100 + [0x18, 0, 42, 0x00]
        r, bc = _opt(bc_in, [OptPass.NOP_ELIM])
        self.assertEqual(len(bc), 4)

    # 58 – unknown opcode > 0x4F (falls through to 1-byte skip)
    def test_unknown_opcode(self):
        r, bc = _opt([0xFE, 0x18, 0, 42, 0x00], [OptPass.HALT_TRUNCATE])
        self.assertEqual(len(bc), 5)  # 0xFE treated as 1-byte; HALT at end

    # 59 – get_bytecode returns list not tuple
    def test_get_bytecode_is_list(self):
        o = PeepholeOptimizer([0x18, 0, 42, 0x00])
        self.assertIsInstance(o.get_bytecode(), list)

    # 60 – get_bytecode after optimization reflects changes
    def test_get_bytecode_after_optimization(self):
        o = PeepholeOptimizer([0x01, 0x18, 0, 42, 0x00])
        o.optimize([OptPass.NOP_ELIM])
        bc = o.get_bytecode()
        self.assertNotIn(0x01, bc)

    # 61 – optimizer does not mutate original
    def test_original_not_mutated(self):
        original = [0x01, 0x18, 0, 42, 0x00]
        o = PeepholeOptimizer(original)
        o.optimize([OptPass.NOP_ELIM])
        self.assertEqual(original, [0x01, 0x18, 0, 42, 0x00])

    # 62 – multiple optimize calls accumulate rules
    def test_multiple_optimize_calls(self):
        o = PeepholeOptimizer([0x01, 0x18, 0, 42, 0x00])
        o.optimize([OptPass.NOP_ELIM])
        o.optimize([OptPass.HALT_TRUNCATE])
        self.assertGreater(len(o.rules_applied), 0)


# ═══════════════════════════════════════════════════════════
# Interaction between dead_code and halt_truncate
# ═══════════════════════════════════════════════════════════
class TestDeadCodeVsHaltTruncate(unittest.TestCase):
    """dead_code truncates after *last* HALT; halt_truncate after *first*."""

    # 63 – halt_truncate is stricter than dead_code
    def test_halt_truncate_stricter(self):
        bc_in = [0x00, 0x18, 0, 42, 0x00]
        r_ht, bc_ht = _opt(list(bc_in), [OptPass.HALT_TRUNCATE])
        r_dc, bc_dc = _opt(list(bc_in), [OptPass.DEAD_CODE])
        self.assertEqual(bc_ht, [0x00])
        self.assertEqual(len(bc_dc), 5)  # keeps everything up to last HALT

    # 64 – dead_code with single trailing byte
    def test_single_trailing_byte(self):
        r, bc = _opt([0x00, 0xAA], [OptPass.DEAD_CODE])
        self.assertEqual(bc, [0x00])
        self.assertEqual(r.savings, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
