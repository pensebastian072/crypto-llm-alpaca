from crypto_llm_alpaca.quality_metrics import QualityMetrics, compute_quality_score


def test_score_blocked_without_gates():
    metrics = QualityMetrics(momentum_gate=False, volume_gate=True)

    assert compute_quality_score(metrics) == 0.0


def test_sentiment_increases_score_when_gates_pass():
    metrics = QualityMetrics(
        sentiment=0.5,
        signal_count_7d=10,
        volume_usd=10_000_000.0,
        momentum_gate=True,
        volume_gate=True,
    )

    assert compute_quality_score(metrics) > 0.0
