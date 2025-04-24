# Copyright 2025
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic import BaseModel, Field


class PlotSavingConfig(BaseModel):
    """Configurations for specifying saving options of the plot."""

    save_html: bool = True
    save_pdf: bool = True
    save_csv: bool = True


class LocalEstimatesPlotConfig(BaseModel):
    """Configurations for specifying parameters of the local estimates plot."""

    pca_n_components: int | None = Field(
        default=50,
        title="Number of PCA components before t-SNE.",
        description="The number of PCA components before t-SNE to use for embedding data preparation.",
    )
    saving: PlotSavingConfig = Field(
        default_factory=PlotSavingConfig,
        title="Configuration for saving the plots.",
        description="Configurations for specifying saving options of the plot.",
    )

    tsne_n_components: int = Field(
        default=2,
        title="Number of t-SNE components",
        description="The number of t-SNE components to use for embedding data preparation.",
    )
    tsne_random_state: int = Field(
        default=42,
        title="Random state for t-SNE",
        description="The random state to use for t-SNE algorithm.",
    )
