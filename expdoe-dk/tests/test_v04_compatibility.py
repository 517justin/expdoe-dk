from pathlib import Path

import expdoe_dk as ed


def test_v04_public_imports_and_checkpoint_remain_readable():
    assert ed.Parameter.__name__ == "Parameter"
    assert ed.LinearConstraint.__name__ == "LinearConstraint"
    assert ed.Space.__name__ == "Space"
    assert ed.Result.__name__ == "Result"
    checkpoint = Path(__file__).parent / "fixtures" / "v04_checkpoint.json"
    campaign = ed.Campaign.load_checkpoint(checkpoint)
    assert campaign.space.objectives == ["yield"]
    assert campaign.history_df()["y"].tolist() == [1.5]


def test_v04_knowledge_payload_round_trips():
    payload = {
        "strict": False,
        "items": [
            {
                "kind": "monotone",
                "param": "x",
                "effect": "increases_objective",
                "n_pairs_per_dim": 5,
                "epsilon": "auto",
                "delta_norm": 0.5,
            }
        ],
    }
    restored = ed.Knowledge.from_dict(payload)
    assert restored.to_dict() == payload
