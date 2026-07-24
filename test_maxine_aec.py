import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parent


class MaxineAecTests(unittest.TestCase):
    def test_graph_preserves_near_far_channel_order(self):
        graph = (ROOT / "pipewire/maxine-aec-filter-chain.conf").read_text()
        self.assertIn('inputs = [ "aec:Near End" "aec:Far End" ]', graph)
        self.assertIn('audio.position = [ FL FR ]', graph)
        self.assertIn('node.name = "zer0_maxine_aec"', graph)
        self.assertIn("node.autoconnect = false", graph)

    def test_full_graph_orders_aec_before_denoise(self):
        graph = (ROOT / "pipewire/maxine-full-filter-chain.conf").read_text()
        self.assertLess(graph.index("name = aec"), graph.index("name = denoise"))
        self.assertIn(
            '{ output = "aec:Output" input = "denoise:Input" }',
            graph,
        )
        self.assertIn('outputs = [ "denoise:Output" ]', graph)

    def test_plugin_uses_nvidia_passing_gpu_selection_and_frame_size(self):
        plugin = (ROOT / "maxine_aec/maxine_aec_ladspa.c").read_text()
        self.assertIn("NVAFX_PARAM_USE_DEFAULT_GPU, 0U", plugin)
        self.assertIn("NVAFX_PARAM_INPUT_SAMPLE_RATE, 48000U", plugin)
        self.assertIn("NVAFX_PARAM_NUM_SAMPLES_PER_INPUT_FRAME, FRAME", plugin)
        self.assertIn("NvAFX_Run(self->effect, inputs, outputs, FRAME, 2U)", plugin)

    def test_link_supervisor_restores_both_exact_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "links"
            pipewire = root / "pipewire"
            pw_link = root / "pw-link"
            probe = root / "probe.py"
            pipewire.write_text("#!/bin/sh\nsleep 0.15\n")
            probe.write_text("raise SystemExit(0)\n")
            pw_link.write_text(
                textwrap.dedent(
                    f"""\
                    #!/bin/sh
                    if [ "$1" = "-i" ]; then
                        printf '%s\\n' capture.zer0_maxine_aec:input_FL capture.zer0_maxine_aec:input_FR
                    else
                        printf '%s -> %s\\n' "$1" "$2" >> {log}
                    fi
                    """
                )
            )
            pipewire.chmod(0o755)
            pw_link.chmod(0o755)
            environment = {
                **os.environ,
                "ZERO_MAXINE_PIPEWIRE_BIN": str(pipewire),
                "ZERO_MAXINE_PW_LINK_BIN": str(pw_link),
                "ZERO_MAXINE_LINK_INTERVAL": "0.01",
                "ZERO_MAXINE_PROBE": str(probe),
                "ZERO_MAXINE_PROBE_INTERVAL": "0.01",
            }
            subprocess.run(
                [str(ROOT / "pipewire/run-maxine-aec")],
                check=True,
                env=environment,
                timeout=2,
            )
            links = log.read_text()
        self.assertIn(
            "Blue_Snowball_797_2020_11_17_99419-00.mono-fallback:capture_MONO"
            " -> capture.zer0_maxine_aec:input_FL",
            links,
        )
        self.assertIn(
            "effect_input.aural_evolution:monitor_FL"
            " -> capture.zer0_maxine_aec:input_FR",
            links,
        )

    def test_stable_supervisor_exits_when_pipewire_child_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipewire = root / "pipewire"
            pactl = root / "pactl"
            probe = root / "probe.py"
            pipewire.write_text("#!/bin/sh\nsleep 0.05\n")
            pactl.write_text(
                "#!/bin/sh\n"
                "if [ \"$1 $2 $3\" = \"list short sources\" ]; then\n"
                "  printf '1\\tzer0_maxine_full\\n'\n"
                "fi\n"
            )
            probe.write_text("raise SystemExit(0)\n")
            pipewire.chmod(0o755)
            pactl.chmod(0o755)
            subprocess.run(
                [str(ROOT / "pipewire/run-maxine-stable")],
                check=True,
                env={
                    **os.environ,
                    "ZERO_MAXINE_PIPEWIRE_BIN": str(pipewire),
                    "ZERO_MAXINE_PACTL_BIN": str(pactl),
                    "ZERO_MAXINE_PROBE": str(probe),
                    "ZERO_MAXINE_PROBE_INTERVAL": "0.01",
                },
                timeout=2,
            )


if __name__ == "__main__":
    unittest.main()
