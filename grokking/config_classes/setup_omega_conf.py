# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (2025) (mail@ruppik.net)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Setup OmegaConf with custom resolvers."""

import omegaconf

from grokking.config_classes.sanitize_dirname import sanitize_dirname


def setup_omega_conf() -> None:
    """Set up OmegaConf with custom resolvers."""
    omegaconf.OmegaConf.register_new_resolver(
        name="sanitize_override_dirname",
        resolver=sanitize_dirname,
    )
