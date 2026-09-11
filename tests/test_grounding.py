from investigator.grounding import ground_findings, quote_in_pool
from investigator.ingest import chunk_text, parse_upload


POLICY = "Customer personal data is retained for 18 months after contract end."


def test_quote_must_come_from_pool():
    assert quote_in_pool("retained for 18 months", [POLICY])
    assert not quote_in_pool("we store data forever", [POLICY])


def test_ground_findings_drops_invented_quotes():
    kept = ground_findings(
        [
            {
                "claim": "Retention is 18 months",
                "quote": "retained for 18 months",
                "document": "policy.txt",
            },
            {
                "claim": "Invented",
                "quote": "we never delete anything",
                "document": "policy.txt",
            },
        ],
        [POLICY],
    )
    assert len(kept) == 1
    assert kept[0].status == "supported"


def test_parse_txt_chunks():
    chunks = parse_upload(
        "vendor_policy.txt",
        POLICY.encode("utf-8"),
        chunk_size=80,
        overlap=10,
    )
    assert chunks
    assert chunks[0]["page"] == 1
    assert chunk_text("a b c", size=2, overlap=0)
