import unittest

from blackboard import DeliberationMesh, Proposal, TurnBoard
from fleet import Horizon, Severity


class Lane:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.snapshots = []

    async def react(self, _text, _context, board):
        self.snapshots.append(board)
        return self.responses.pop(0) if self.responses else None


class BlackboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_lanes_see_and_can_steer_peer_results(self):
        live = Proposal("live", "spoken", "Initial answer", 0.6, Horizon.INSTANT)
        correction = Proposal(
            "pro",
            "spoken",
            "Corrected answer",
            0.95,
            Horizon.META,
            Severity.INTERRUPT,
            supersedes=live.id,
            caused_by=(live.id,),
        )
        live_lane = Lane("live", [live, None])
        pro_lane = Lane("pro", [None, correction])
        board = await DeliberationMesh([live_lane, pro_lane], max_rounds=3).run(
            "question"
        )

        self.assertEqual(board.best("spoken").text, "Corrected answer")
        self.assertIn(live, pro_lane.snapshots[1])
        self.assertIn(correction, live_lane.snapshots[2])

    async def test_round_bound_stops_feedback_loops(self):
        class EchoLane:
            name = "echo"

            def __init__(self):
                self.calls = 0

            async def react(self, _text, _context, board):
                self.calls += 1
                return Proposal(
                    self.name,
                    "echo",
                    str(self.calls),
                    0.5,
                    Horizon.SHORT,
                    caused_by=tuple(item.id for item in board),
                )

        lane = EchoLane()
        board = await DeliberationMesh([lane], max_rounds=3).run("loop")
        self.assertEqual(lane.calls, 3)
        self.assertEqual(board.version, 3)

    def test_unknown_supersession_is_rejected(self):
        board = TurnBoard()
        with self.assertRaises(ValueError):
            board.publish(
                Proposal(
                    "high",
                    "plan",
                    "replacement",
                    0.9,
                    Horizon.MID,
                    supersedes="missing",
                )
            )

    def test_cross_lane_supersede_emits_explicit_steer_edge(self):
        edges: list[tuple[str, str, str]] = []
        board = TurnBoard(
            on_steer=lambda steering, steered, topic: edges.append(
                (steering, steered, topic)
            )
        )
        live = Proposal("live", "spoken", "Initial answer", 0.6, Horizon.INSTANT)
        board.publish(live)
        correction = Proposal(
            "high",
            "spoken",
            "Corrected answer",
            0.9,
            Horizon.MID,
            supersedes=live.id,
        )
        board.publish(correction)
        self.assertEqual(edges, [("high", "live", "spoken")])

        # A lane refining its own proposal is not steering another lane.
        board.publish(
            Proposal(
                "high",
                "spoken",
                "Refined answer",
                0.95,
                Horizon.MID,
                supersedes=correction.id,
            )
        )
        self.assertEqual(len(edges), 1)


if __name__ == "__main__":
    unittest.main()
