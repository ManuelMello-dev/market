import unittest

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


if __name__ == "__main__":
    unittest.main()
