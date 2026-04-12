"""Pytest test suite for flux-optimizer."""
import pytest
from optimizer import (
    PeepholeOptimizer, OptPass, OptResult
)


# ── Fixtures ──

@pytest.fixture
def simple_movi():
    """MOVI R0, 42; HALT"""
    return [0x18, 0, 42, 0x00]


@pytest.fixture
def movi_addi():
    """MOVI R0, 10; ADDI R0, 20; HALT — should constant-fold to MOVI R0, 30"""
    return [0x18, 0, 10, 0x19, 0, 20, 0x00]


@pytest.fixture
def nop_bytecode():
    """NOP; MOVI R0, 42; HALT"""
    return [0x01, 0x18, 0, 42, 0x00]


@pytest.fixture
def halt_truncate_bc():
    """MOVI R0, 42; HALT; dead code after HALT"""
    return [0x18, 0, 42, 0x00, 0x18, 1, 10, 0x00]


@pytest.fixture
def redundant_mov_bc():
    """MOV R0, R0, 0; HALT — redundant mov to same register"""
    return [0x3A, 0, 0, 0, 0x00]


@pytest.fixture
def strength_reduce_bc():
    """MOVI R1, 2; MUL R0, R0, R1; HALT — strength reduce to ADD"""
    return [0x18, 1, 2, 0x22, 0, 0, 1, 0x00]


# ── OptResult tests ──

class TestOptResult:
    def test_savings_computed_on_init(self):
        result = OptResult(original_bytes=10, optimized_bytes=7, passes_run=["nop_elim"])
        assert result.savings == 3

    def test_zero_savings(self):
        result = OptResult(original_bytes=5, optimized_bytes=5, passes_run=[])
        assert result.savings == 0

    def test_negative_savings_when_optimized_larger(self):
        """Optimization could theoretically increase size."""
        result = OptResult(original_bytes=5, optimized_bytes=6, passes_run=[])
        assert result.savings == -1

    def test_rules_applied_default_empty(self):
        result = OptResult(original_bytes=5, optimized_bytes=5, passes_run=[])
        assert result.rules_applied == []


# ── HALT truncate ──

class TestHaltTruncate:
    def test_removes_code_after_halt(self, halt_truncate_bc):
        opt = PeepholeOptimizer(halt_truncate_bc)
        result = opt.optimize([OptPass.HALT_TRUNCATE])
        assert len(opt.get_bytecode()) == 4

    def test_records_rule(self, halt_truncate_bc):
        opt = PeepholeOptimizer(halt_truncate_bc)
        result = opt.optimize([OptPass.HALT_TRUNCATE])
        assert any("halt_truncate" in r for r in result.rules_applied)

    def test_no_change_if_no_code_after_halt(self, simple_movi):
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize([OptPass.HALT_TRUNCATE])
        assert result.savings == 0

    def test_empty_bytecode(self):
        opt = PeepholeOptimizer([])
        result = opt.optimize([OptPass.HALT_TRUNCATE])
        assert len(opt.get_bytecode()) == 0


# ── NOP elimination ──

class TestNopElim:
    def test_removes_nop(self, nop_bytecode):
        opt = PeepholeOptimizer(nop_bytecode)
        result = opt.optimize([OptPass.NOP_ELIM])
        assert len(opt.get_bytecode()) == 4  # removed 1 NOP

    def test_multiple_nops(self):
        bc = [0x01, 0x01, 0x18, 0, 42, 0x00]
        opt = PeepholeOptimizer(bc)
        result = opt.optimize([OptPass.NOP_ELIM])
        assert len(opt.get_bytecode()) == 4

    def test_no_nops(self, simple_movi):
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize([OptPass.NOP_ELIM])
        assert result.savings == 0


# ── Constant folding ──

class TestConstantFold:
    def test_fold_movi_addi(self, movi_addi):
        opt = PeepholeOptimizer(movi_addi)
        result = opt.optimize([OptPass.CONSTANT_FOLD])
        bc = opt.get_bytecode()
        assert bc[0] == 0x18  # MOVI
        assert bc[2] == 30    # combined value

    def test_fold_negative(self):
        # MOVI R0, 10; ADDI R0, -5 -> MOVI R0, 5
        opt = PeepholeOptimizer([0x18, 0, 10, 0x19, 0, 0xFB, 0x00])
        result = opt.optimize([OptPass.CONSTANT_FOLD])
        bc = opt.get_bytecode()
        assert bc[2] == 5

    def test_no_fold_different_registers(self):
        # MOVI R0, 10; ADDI R1, 20 — different registers, no fold
        opt = PeepholeOptimizer([0x18, 0, 10, 0x19, 1, 20, 0x00])
        result = opt.optimize([OptPass.CONSTANT_FOLD])
        assert len(opt.get_bytecode()) == 7  # unchanged

    def test_overflow_not_folded(self):
        # MOVI R0, 120; ADDI R0, 50 = 170 > 127, should not fold
        opt = PeepholeOptimizer([0x18, 0, 120, 0x19, 0, 50, 0x00])
        result = opt.optimize([OptPass.CONSTANT_FOLD])
        assert len(opt.get_bytecode()) == 7


# ── Strength reduction ──

class TestStrengthReduce:
    def test_mul_by_2_reduces_to_add(self, strength_reduce_bc):
        opt = PeepholeOptimizer(strength_reduce_bc)
        result = opt.optimize([OptPass.STRENGTH_REDUCE])
        bc = opt.get_bytecode()
        assert 0x20 in bc  # ADD opcode
        assert any("strength_reduce" in r for r in result.rules_applied)

    def test_no_reduce_mul_by_3(self):
        # MOVI R1, 3; MUL R0, R0, R1 — not 2, no reduce
        bc = [0x18, 1, 3, 0x22, 0, 0, 1, 0x00]
        opt = PeepholeOptimizer(bc)
        result = opt.optimize([OptPass.STRENGTH_REDUCE])
        assert not any("strength_reduce" in r for r in result.rules_applied)


# ── Redundant MOV ──

class TestRedundantMov:
    def test_removes_redundant_mov(self, redundant_mov_bc):
        opt = PeepholeOptimizer(redundant_mov_bc)
        result = opt.optimize([OptPass.REDUNDANT_MOV])
        bc = opt.get_bytecode()
        assert 0x3A not in bc[:-1]  # MOV removed

    def test_keeps_non_redundant_mov(self):
        bc = [0x3A, 1, 0, 0, 0x00]  # MOV R1, R0 — not redundant
        opt = PeepholeOptimizer(bc)
        result = opt.optimize([OptPass.REDUNDANT_MOV])
        assert result.savings == 0


# ── Combined passes ──

class TestCombinedPasses:
    def test_all_passes(self, simple_movi):
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize()
        assert result.savings == 0

    def test_multi_pattern(self, nop_bytecode):
        """NOP + dead code after HALT."""
        bc = [0x01, 0x18, 0, 42, 0x00, 0x18, 1, 10]
        opt = PeepholeOptimizer(bc)
        result = opt.optimize([OptPass.NOP_ELIM, OptPass.HALT_TRUNCATE])
        assert result.savings > 0

    @pytest.mark.parametrize("passes", [
        [OptPass.NOP_ELIM],
        [OptPass.HALT_TRUNCATE],
        [OptPass.CONSTANT_FOLD],
        [OptPass.STRENGTH_REDUCE],
        [OptPass.REDUNDANT_MOV],
    ])
    def test_individual_passes_dont_crash(self, simple_movi, passes):
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize(passes)
        assert isinstance(result, OptResult)

    def test_dead_code_pass_is_noop(self, simple_movi):
        """Dead code pass is a stub."""
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize([OptPass.DEAD_CODE])
        assert result.savings == 0

    def test_peephole_pass_is_identity(self, simple_movi):
        """Peephole pass is a stub."""
        opt = PeepholeOptimizer(simple_movi)
        result = opt.optimize([OptPass.PEEPHOLE])
        assert result.savings == 0


# ── Bytecode output ──

class TestBytecodeOutput:
    def test_get_bytecode_returns_list(self, simple_movi):
        opt = PeepholeOptimizer(simple_movi)
        assert isinstance(opt.get_bytecode(), list)

    def test_original_not_mutated(self, simple_movi):
        original = list(simple_movi)
        opt = PeepholeOptimizer(simple_movi)
        opt.optimize([OptPass.NOP_ELIM])
        assert opt.original == original
