import base64
import unittest

from src.engine import ENGINE_VERSION, analyze_request
from src.schemas import AnalyzeRequest


class SponsoredTrialAnalyticsEngineTest(unittest.TestCase):
    def sample_payload(self):
        return {
            "trial": {
                "sponsored_trial_id": 77,
                "title": "Magnesium Sleep Trial",
                "sponsor_name": "Acme",
                "protocol_hash": "abc123",
                "primary_endpoint": "sleep_score",
            },
            "endpoints": [
                {"key": "sleep_score", "label": "Sleep score", "direction": "higher_better", "role": "primary", "responder_threshold": 4},
                {"key": "rhr_bpm", "label": "Resting heart rate", "direction": "lower_better", "role": "secondary"},
            ],
            "participants": [
                {
                    "participant_id": "p1",
                    "status": "completed",
                    "verified_compliance_pct": 92,
                    "subgroups": {"device": "oura"},
                    "endpoints": {
                        "sleep_score": {"baseline_values": [70, 72], "intervention_values": [80, 82], "daily_values": [76, 78, 82]},
                        "rhr_bpm": {"baseline_values": [60, 61], "intervention_values": [57, 58], "daily_values": [59, 58, 57]},
                    },
                },
                {
                    "participant_id": "p2",
                    "status": "completed",
                    "verified_compliance_pct": 80,
                    "subgroups": {"device": "oura"},
                    "endpoints": {
                        "sleep_score": {"baseline_values": [65, 66], "intervention_values": [70, 72], "daily_values": [67, 70, 72]},
                        "rhr_bpm": {"baseline_values": [65, 64], "intervention_values": [63, 62], "daily_values": [64, 63, 62]},
                    },
                },
                {
                    "participant_id": "p3",
                    "status": "withdrawn",
                    "verified_compliance_pct": 40,
                    "subgroups": {"device": "whoop"},
                    "endpoints": {"sleep_score": {"baseline_values": [74], "intervention_values": [73], "daily_values": [73]}},
                },
            ],
            "concurrent_interventions": [{"category": "caffeine"}],
            "adverse_events": [{"severity": "mild"}],
        }

    def test_engine_returns_versioned_report_pdf_and_hash(self):
        report = analyze_request(AnalyzeRequest.model_validate(self.sample_payload()))
        self.assertEqual(ENGINE_VERSION, report["engine_version"])
        self.assertRegex(report["report_hash"], r"^[0-9a-f]{64}$")
        self.assertGreater(len(base64.b64decode(report["pdf_base64"])), 1000)
        self.assertEqual(2, len(report["endpoint_results"]))

    def test_primary_endpoint_has_itt_per_protocol_and_bonferroni(self):
        report = analyze_request(AnalyzeRequest.model_validate(self.sample_payload()))
        primary = report["endpoint_results"][0]
        self.assertEqual("sleep_score", primary["endpoint_key"])
        self.assertEqual(3, primary["n_itt"])
        self.assertEqual(2, primary["n_per_protocol"])
        self.assertGreater(primary["itt_mean_delta"], 0)
        self.assertIsNotNone(primary["bonferroni_p"])
        self.assertEqual(2, primary["responders"])

    def test_lower_better_endpoint_orients_delta_positive_when_value_drops(self):
        report = analyze_request(AnalyzeRequest.model_validate(self.sample_payload()))
        rhr = report["endpoint_results"][1]
        self.assertEqual("rhr_bpm", rhr["endpoint_key"])
        self.assertGreater(rhr["itt_mean_delta"], 0)

    def test_paired_t_and_bh_fdr_fields_are_reported(self):
        payload = self.sample_payload()
        payload["participants"] = []
        for idx in range(30):
            baseline = 70 + (idx % 5)
            payload["participants"].append(
                {
                    "participant_id": f"p{idx}",
                    "status": "completed",
                    "verified_compliance_pct": 90,
                    "endpoints": {
                        "sleep_score": {
                            "baseline_values": [baseline],
                            "intervention_values": [baseline + 4.5 + (idx % 3) * 0.2],
                            "daily_values": [],
                        }
                    },
                }
            )

        report = analyze_request(AnalyzeRequest.model_validate(payload))
        primary = report["endpoint_results"][0]

        self.assertAlmostEqual(4.7, primary["itt_mean_delta"], places=3)
        self.assertGreater(primary["intervention_mean"], primary["baseline_mean"])
        self.assertIsNotNone(primary["itt_delta_sd"])
        self.assertLess(primary["paired_t_p"], 0.05)
        self.assertLess(primary["bh_fdr_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
