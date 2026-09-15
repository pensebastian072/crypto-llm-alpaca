from crypto_llm_alpaca.llm import MockLLMProvider
from crypto_llm_alpaca.news import parse_rss


def test_parse_rss_maps_symbols_and_dedupes_hash():
    xml = """<rss><channel><item><title>Bitcoin rally accelerates</title><link>https://example.com/a</link><description>BTC inflow improves.</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_rss(xml, "fixture", ("BTC/USD", "ETH/USD"))
    assert len(items) == 1
    assert items[0].symbols == ("BTC/USD",)
    assert len(items[0].dedupe_hash) == 64


def test_mock_llm_outputs_valid_bounded_signal():
    xml = """<rss><channel><item><title>Ethereum hack fears fade</title><link>https://example.com/e</link><description>ETH rally follows.</description><pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate></item></channel></rss>"""
    item = parse_rss(xml, "fixture", ("ETH/USD",))[0]
    signal = MockLLMProvider().extract(item, "ETH/USD")
    assert -1 <= signal.sentiment <= 1
    assert 0 <= signal.confidence <= 1
    assert signal.symbol == "ETH/USD"
