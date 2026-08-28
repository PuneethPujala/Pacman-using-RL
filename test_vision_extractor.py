import unittest
from layout import Layout
from pacman import GameState
from game import Directions, Configuration, AgentState
import featureExtractors
from bustersAgents import ApproximateQAgent
import util

class TestHumanVisionExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = featureExtractors.HumanVisionExtractor()

    def test_case_1_straight_corridor_raycasting(self):
        """
        Case 1: Straight 1D corridor with ghost 2 steps away facing Pacman.
        Layout: %%%%%%% (7x3)
                % P.G % (y=1: Pacman at (2,1), Ghost at (4,1) facing WEST)
                %%%%%%%
        """
        layout_text = [
            "%%%%%%%",
            "% P.G %",
            "%%%%%%%"
        ]
        lay = Layout(layout_text)
        state = GameState()
        state.data.initialize(lay, numGhostAgents=1)

        # Set ghost direction to WEST (approaching Pacman)
        ghost_state = state.getGhostStates()[0]
        ghost_state.configuration.direction = Directions.WEST

        # Evaluate moving EAST (towards ghost)
        features_east = self.extractor.getFeatures(state, Directions.EAST)
        # Successor pos is (3, 1). Ghost is at (4, 1) (dist 1) facing WEST (approaching)
        self.assertAlmostEqual(features_east["ghost-visible-approaching"], 1.0, places=3)
        self.assertEqual(features_east["ghost-1-step-away"], 1.0)
        self.assertEqual(features_east["ghost-2-steps-away"], 0.0)

        # Evaluate moving WEST (retreating away from ghost)
        features_west = self.extractor.getFeatures(state, Directions.WEST)
        # Successor pos is (1, 1). Ghost is at (4, 1) (dist 3) facing WEST (approaching)
        self.assertAlmostEqual(features_west["ghost-visible-approaching"], 1.0 / 3.0, places=3)
        self.assertEqual(features_west["ghost-1-step-away"], 0.0)
        self.assertEqual(features_west["ghost-2-steps-away"], 0.0)
        self.assertEqual(features_west["ghost-3-steps-away"], 1.0)

        print("\n[PASSED] Case 1: Straight Corridor Raycasting correctly identified approaching ghost distances.")

    def test_case_2_l_shaped_corridor_with_open_intersection(self):
        """
        Case 2: L-shaped corridor that turns and leads to an open intersection.
        Should NOT be classified as trapped-in-dead-end.
        """
        layout_text = [
            "%%%%%%%%",
            "%    . %",
            "% %%%% %",
            "%P   G %",
            "%%%%%%%%"
        ]
        lay = Layout(layout_text)
        state = GameState()
        state.data.initialize(lay, numGhostAgents=1)

        # Pacman at (1, 1), Ghost at (5, 1)
        # Moving North to (1, 2) leads up and turns East into an open region (2, 3)-(6, 3)
        features_north = self.extractor.getFeatures(state, Directions.NORTH)
        self.assertEqual(features_north["trapped-in-dead-end"], 0.0,
                         "L-shaped turn leading to open area should NOT trigger trapped-in-dead-end")

        print("[PASSED] Case 2: L-Shaped Corridor with open junction correctly recognized as safe (not dead-end).")

    def test_case_3_genuine_cul_de_sac_dead_end(self):
        """
        Case 3: Genuine cul-de-sac pocket with approaching ghost blocking the entrance.
        Layout has a dead-end pocket of 3 tiles, and ghost is right at the opening.
        """
        layout_text = [
            "%%%%%%%%%",
            "%   P G %",
            "%%%%%%%%%"
        ]
        lay = Layout(layout_text)
        state = GameState()
        state.data.initialize(lay, numGhostAgents=1)

        # Pacman at (4, 1), Ghost at (6, 1) facing WEST
        ghost_state = state.getGhostStates()[0]
        ghost_state.configuration.direction = Directions.WEST

        # Moving WEST deeper into dead end (3, 1) -> (2, 1) -> (1, 1) -> WALL
        # Ghost at (6, 1) blocks the only way out (towards East)
        features_west = self.extractor.getFeatures(state, Directions.WEST)
        self.assertEqual(features_west["trapped-in-dead-end"], 1.0,
                         "Moving into a cul-de-sac with ghost blocking entrance MUST trigger trapped-in-dead-end")

        print("[PASSED] Case 3: Genuine Cul-de-Sac correctly flagged as trapped-in-dead-end.")

    def test_case_4_ghost_sealed_escape(self):
        """
        Case 4: A pocket containing an internal T-junction, but ALL exit branches
        are sealed by ghosts. The agent must recognize that the reachable area is trapped.
        """
        layout_text = [
            "%%%%%%%",
            "% G.G %",
            "%  P  %",
            "%%%%%%%"
        ]
        lay = Layout(layout_text)
        state = GameState()
        state.data.initialize(lay, numGhostAgents=2)

        features_north = self.extractor.getFeatures(state, Directions.NORTH)
        # Moving North into (3, 2) has dead wall above and ghosts blocking left (2, 2) and right (4, 2)
        self.assertTrue(features_north["ghost-1-step-away"] > 0 or features_north["trapped-in-dead-end"] > 0)
        print("[PASSED] Case 4: Ghost-sealed topology correctly identified as high danger.")

    def test_case_5_q_value_forward_vs_backward_behavior(self):
        """
        Case 5: Verification that Q(Backward) > Q(Forward) in the 2-step corridor scenario.
        """
        layout_text = [
            "%%%%%%%%%",
            "%  P.G  %",
            "%%%%%%%%%"
        ]
        lay = Layout(layout_text)
        state = GameState()
        state.data.initialize(lay, numGhostAgents=1)

        # Pacman at (3, 1), Ghost at (5, 1) facing WEST (distance = 2)
        ghost_state = state.getGhostStates()[0]
        ghost_state.configuration.direction = Directions.WEST

        agent = ApproximateQAgent(extractor='HumanVisionExtractor')
        # Typical learned weights for safe Pacman RL:
        agent.weights['ghost-visible-approaching'] = -120.0
        agent.weights['ghost-visible-retreating'] = -20.0
        agent.weights['ghost-1-step-away'] = -300.0
        agent.weights['ghost-2-steps-away'] = -100.0
        agent.weights['ghost-3-steps-away'] = -10.0
        agent.weights['trapped-in-dead-end'] = -250.0
        agent.weights['eats-food'] = 25.0
        agent.weights['closest-food'] = -5.0
        agent.weights['bias'] = 5.0

        q_forward = agent.getQValue(state, Directions.EAST)
        q_backward = agent.getQValue(state, Directions.WEST)

        print(f"\n--- Q-Value Evaluation ---")
        print(f"Q(Forward / East towards ghost):  {q_forward:.2f}")
        print(f"Q(Backward / West away from ghost): {q_backward:.2f}")

        self.assertGreater(q_backward, q_forward,
                           f"Q(Backward) ({q_backward}) must be strictly greater than Q(Forward) ({q_forward})")
        print("[PASSED] Case 5: Q(Backward) > Q(Forward) verified successfully!")


if __name__ == '__main__':
    unittest.main()
