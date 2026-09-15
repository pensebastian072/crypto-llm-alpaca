from crypto_llm_alpaca.attribution import reconcile_symbols


def test_aave_article_overrides_bad_btc_output():
    result = reconcile_symbols(
        ("BTC/USD",),
        ("AAVE/USD",),
        ("AAVE/USD", "UNI/USD", "DOT/USD"),
    )

    assert result.symbols == ("AAVE/USD",)
    assert result.corrected is True


def test_empty_llm_output_falls_back_to_article_symbol():
    result = reconcile_symbols(
        (),
        ("UNI/USD",),
        ("AAVE/USD", "UNI/USD"),
    )

    assert result.symbols == ("UNI/USD",)
    assert result.fallback is True


def test_mixed_output_prefers_article_overlap():
    result = reconcile_symbols(
        ("BTC/USD", "DOT/USD"),
        ("DOT/USD",),
        ("BTC/USD", "DOT/USD"),
    )

    assert result.symbols == ("DOT/USD",)


def test_invalid_symbols_fall_back_to_article_symbol():
    result = reconcile_symbols(
        ("DOGE/USD",),
        ("AAVE/USD",),
        ("AAVE/USD", "UNI/USD"),
    )

    assert result.symbols == ("AAVE/USD",)
    assert result.fallback is True


def test_primary_symbol_bias_uses_first_article_symbol():
    result = reconcile_symbols(
        ("ARB/USD",),
        ("AAVE/USD", "ARB/USD"),
        ("AAVE/USD", "ARB/USD", "BTC/USD"),
    )

    assert result.symbols == ("AAVE/USD",)
    assert result.corrected is True
