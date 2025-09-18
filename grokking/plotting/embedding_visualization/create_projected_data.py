"""Functions to create a projection of the input data using PCA and t-SNE."""

import logging

import numpy as np
import sklearn.decomposition
import sklearn.manifold

from grokking.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_projected_data(
    array: np.ndarray,
    pca_n_components: int | None = 50,
    tsne_n_components: int = 2,
    tsne_random_state: int = 42,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> np.ndarray:
    """Create a projection of the input data using PCA and t-SNE."""
    # Apply PCA if requested
    if pca_n_components:
        if verbosity >= Verbosity.VERBOSE:
            logger.info(
                msg=f"Applying PCA to reduce the number of dimensions to {pca_n_components = } ...",  # noqa: G004 - low overhead
            )
        pca = sklearn.decomposition.PCA(
            n_components=pca_n_components,
        )
        array = pca.fit_transform(
            X=array,
        )
        if verbosity >= Verbosity.VERBOSE:
            logger.info(
                msg=f"Applying PCA to reduce the number of dimensions to {pca_n_components = } DONE",  # noqa: G004 - low overhead
            )

    # Apply t-SNE
    if verbosity >= Verbosity.VERBOSE:
        logger.info(
            msg=f"Applying t-SNE to reduce the number of dimensions to {tsne_n_components = } ...",  # noqa: G004 - low overhead
        )
    tsne = sklearn.manifold.TSNE(
        n_components=tsne_n_components,
        random_state=tsne_random_state,
    )
    tsne_array: np.ndarray = tsne.fit_transform(
        X=array,
    )
    if verbosity >= Verbosity.VERBOSE:
        logger.info(
            msg=f"Applying t-SNE to reduce the number of dimensions to {tsne_n_components = } DONE",  # noqa: G004 - low overhead
        )

    return tsne_array
