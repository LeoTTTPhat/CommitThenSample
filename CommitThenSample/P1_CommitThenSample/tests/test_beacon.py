"""DESIGN §7.4 test_beacon: t(r) formula; randomness = SHA-256(sig); chain links; epoch alignment."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from ctsample.protocol import beacon as bc


@pytest.fixture(scope="module")
def drand():
    return bc.DrandRounds()


def test_loaded_chain_parameters(drand):
    assert len(drand) == 3000
    assert drand.first_round == 2570006 and drand.last_round == 2573005
    assert drand.genesis == 1595431050 and drand.period == 30
    assert drand.chain_hash == bc.CHAIN_HASH_DEFAULT
    assert drand.scheme == "pedersen-bls-chained"


def test_round_time_formula(drand):
    assert drand.t(2570006) == 1672531200             # 2023-01-01T00:00:00Z (MANIFEST)
    assert drand.t(2573005) == 1672621170             # 2023-01-02T00:59:30Z
    assert drand.t(1) == drand.genesis
    for r in (2570006, 2571000, 2573005):
        assert drand.t(r) == drand.genesis + (r - 1) * 30
        assert drand.round_at(drand.t(r)) == r
        assert drand.round_at(drand.t(r) + 29.9) == r


def test_randomness_and_chain_all_rounds(drand):
    rep = drand.check_all()
    assert rep["ok"], rep
    assert rep["n"] == 3000 and not rep["bad_randomness"] and not rep["bad_links"] and not rep["gaps"]
    r = 2571234
    b = drand.get(r)
    assert hashlib.sha256(b.signature).digest() == b.randomness
    assert drand.check_chain(r) is True
    assert drand.check_chain(drand.first_round) is None          # predecessor not loaded


def test_tampered_rounds_are_detected(drand):
    b = drand.get(2570100)
    sig = bytearray(b.signature); sig[5] ^= 1
    bad = dataclasses.replace(b, signature=bytes(sig))
    assert not bc.check_round_randomness(bad)
    prev = drand.get(2570099)
    assert bc.check_round_link(prev, b)
    # the successor of a forged round no longer links to it
    nxt = drand.get(2570101)
    assert bc.check_round_link(b, nxt) and not bc.check_round_link(bad, nxt)
    bad2 = dataclasses.replace(b, previous_signature=bytes(96))
    assert not bc.check_round_link(prev, bad2)
    assert not bc.check_round_link(drand.get(2570098), b)          # wrong predecessor
    # a small in-memory chain with a broken link
    rs = [drand.get(r) for r in range(2570006, 2570016)]
    rs[4] = dataclasses.replace(rs[4], previous_signature=bytes(96))
    d2 = bc.DrandRounds(rounds=rs, genesis=drand.genesis, period=30)
    rep = d2.check_all()
    assert not rep["ok"] and rep["bad_links"] == [2570010]


def test_bls_stub(drand):
    with pytest.raises(NotImplementedError):
        drand.verify_bls(2570006)
    m = drand.get(2570007).chained_message()
    assert len(m) == 32


def test_epoch_alignment(drand):
    clk = bc.align_stream(drand, 20)
    assert clk.r0 == drand.first_round
    for e in range(20):
        assert clk.r_star(e) == clk.r0 + e + 1
        assert clk.t_e(e) == drand.t(clk.r0 + e) - 3.0
        assert clk.t_e_us(e) == drand.t(clk.r0 + e) * 10 ** 6 - 3_000_000
        assert clk.t_e(e + 1) - clk.t_e(e) == 30
        # t(r*_e) = t_{e+1} + Δc
        assert clk.t_rstar(e) == clk.t_e(e + 1) + 3.0
        assert clk.deadline(e) == clk.t_rstar(e) - 0.3 - 1.0
        assert clk.epoch_of(clk.t_e(e)) == e
        assert clk.epoch_of(clk.t_e(e) + 29.999) == e
        assert clk.r_star(e) in drand and clk.r_star(e) - 1 in drand


def test_max_epochs_and_limits(drand):
    assert bc.max_epochs(drand, ahead=1) == 2999
    assert bc.max_epochs(drand, ahead=2) == 2998           # ≤ 2,998 holds for both options
    clk = bc.align_stream(drand, 2999)
    assert clk.r_star(2998) == drand.last_round
    with pytest.raises(ValueError):
        bc.align_stream(drand, 3000)
    with pytest.raises(ValueError):
        bc.align_stream(drand, 2999, ahead=2)
    clk2 = bc.align_stream(drand, 2998, ahead=2)
    assert clk2.r_star(0) == clk2.r0 + 2


def test_untrusted_parsing_rejects_malformed(tmp_path):
    (tmp_path / "chain_info.json").write_text(
        '{"public_key": "' + "00" * 48 + '", "period": 30, "genesis_time": 1595431050, "hash": "x", "schemeID": "s"}')
    (tmp_path / "rounds.jsonl").write_text('{"round": 5, "randomness": "00", "signature": "", "previous_signature": ""}\n')
    with pytest.raises(ValueError):
        bc.DrandRounds(tmp_path)
