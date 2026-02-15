import unittest
from unittest import mock

from universal_mind import UniversalCognitiveCore


class UniversalMindTests(unittest.TestCase):
    def setUp(self):
        self.mind = UniversalCognitiveCore("test")
        self.obs = {
            "open": 1.0,
            "high": 1.2,
            "low": 0.8,
            "close": 1.1,
            "volume": 10,
            "timestamp": "t",
        }

    def test_strengthen_concept_increments_confidence(self):
        first = self.mind.ingest(self.obs, domain="finance")
        concept_id = first["concept_formed"]

        initial_confidence = self.mind.concepts[concept_id].confidence
        second = self.mind.ingest(self.obs, domain="finance")
        self.assertEqual(second["concept_formed"], concept_id)
        self.assertGreater(self.mind.concepts[concept_id].confidence, initial_confidence)
        self.assertLessEqual(self.mind.concepts[concept_id].confidence, 1.0)

    def test_stale_signature_mapping_recreates_concept(self):
        first = self.mind.ingest(self.obs, domain="finance")
        concept_id = first["concept_formed"]
        signature = next(iter(self.mind.concept_signatures.keys()))

        # Simulate stale mapping by removing concept but keeping signature mapping
        del self.mind.concepts[concept_id]
        self.mind.concept_signatures[signature] = concept_id

        rebuilt = self.mind.ingest(self.obs, domain="finance")
        new_concept_id = rebuilt["concept_formed"]

        self.assertNotEqual(concept_id, new_concept_id)
        self.assertIn(signature, self.mind.concept_signatures)
        self.assertEqual(self.mind.concept_signatures[signature], new_concept_id)
        self.assertEqual(len(self.mind.concepts), 1)


class StreamMarketDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_skips_duplicate_timestamps(self):
        from universal_mind import stream_market_data

        class DummyMind:
            def __init__(self):
                self.ingest_calls = []

            def ingest(self, obs, domain="finance"):
                self.ingest_calls.append(obs)
                return {
                    "concept_formed": "c",
                    "new_rules": 0,
                    "current_concepts": 1,
                    "urgency": "normal",
                }

            def introspect(self):
                return {}

        responses = [
            {"values": [{"datetime": "t1", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]},
            {"values": [{"datetime": "t1", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]},
            {"values": [{"datetime": "t2", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]},
        ]

        async def fake_fetch_market_data(symbol, api_key, interval):
            return responses.pop(0)

        async def fake_transform_market_data(raw_data):
            return {"domain": "finance", "timestamp": raw_data.get("datetime")}

        mind = DummyMind()

        with mock.patch("universal_mind.fetch_market_data", side_effect=fake_fetch_market_data), \
             mock.patch("universal_mind.transform_market_data", side_effect=fake_transform_market_data):
            await stream_market_data(
                symbol="AAPL",
                api_key="k",
                interval="1min",
                delay_seconds=0,
                mind_instance=mind,
                max_iterations=3,
            )

        self.assertEqual(len(mind.ingest_calls), 2)
        self.assertEqual([obs["timestamp"] for obs in mind.ingest_calls], ["t1", "t2"])


if __name__ == "__main__":
    unittest.main()
