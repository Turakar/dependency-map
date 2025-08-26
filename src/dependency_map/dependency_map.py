import functools
from typing import Generic, Protocol, Self, TypeVar

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import log_softmax, softmax
from tqdm import tqdm

from dependency_map.sequence_logo import SequenceLogo
from dependency_map.util import matplotlib_scale_as_plotly

# having this type be invariant is correct and useful, because it is used
# as both a return type and argument
TokenizedType = TypeVar("TokenizedType")


class TokenizeFunc(Protocol, Generic[TokenizedType]):  # pyright: ignore[reportInvalidTypeVarUse]
    def __call__(self, sequence: str, mask: int | None) -> TokenizedType: ...


class ForwardFunc(Protocol, Generic[TokenizedType]):  # pyright: ignore[reportInvalidTypeVarUse]
    def __call__(self, batch: list[TokenizedType]) -> np.ndarray: ...


class DependencyMap:
    def __init__(
        self, sequence: str, dependency_map: np.ndarray, reconstruction: np.ndarray
    ) -> None:
        """
        Initialize a DependencyMap object directly.

        Args:
            sequence: The input DNA sequence.
            dependency_map: The computed dependency map as a numpy array.
            reconstruction: The reconstruction probabilities as a numpy array.
        """
        self.sequence = sequence
        self.dependency_map = dependency_map
        self.reconstruction = reconstruction

    @staticmethod
    def make_dataset(
        sequence: str,
        tokenize_func: TokenizeFunc[TokenizedType],
        subset: tuple[int, int] | None = None,
    ) -> list[TokenizedType]:
        """
        Create a dataset of tokenized sequences for dependency map analysis.

        Args:
            sequence: The input DNA sequence.
            tokenize_func: Function to tokenize the sequence.
            subset: Optional tuple specifying the start and end indices for a subsequence.

        Returns:
            A list of tokenized sequences including reference, masked, and mutated variants.
        """
        _check_sequence(sequence)
        tokenized = [tokenize_func(sequence, mask=None)]
        if subset is None:
            start, end = 0, len(sequence)
        else:
            start, end = subset
        for i in range(start, end):
            tokenized.append(tokenize_func(sequence, mask=i))
        for i in range(start, end):
            for nt in "ACGT":
                if sequence[i] != nt:
                    mutated = list(sequence)
                    mutated[i] = nt
                    tokenized.append(tokenize_func("".join(mutated), mask=None))
        return tokenized

    @staticmethod
    def make_dataset_str(
        sequence: str, mask_char: str = "X", subset: tuple[int, int] | None = None
    ) -> list[str]:
        """
        Create a dataset of string sequences with masked characters for dependency map analysis.

        Args:
            sequence: The input DNA sequence.
            mask_char: Character to use for masking.
            subset: Optional tuple specifying the start and end indices for a subsequence.

        Returns:
            A list of string sequences including reference, masked, and mutated variants.
        """
        return DependencyMap.make_dataset(
            sequence, functools.partial(_str_tokenize_func, mask_char=mask_char), subset=subset
        )

    @classmethod
    def from_logits(cls: type[Self], sequence: str, logits: np.ndarray) -> Self:
        """
        Construct a DependencyMap from model logits.

        Args:
            sequence: The input DNA sequence.
            logits: Model output logits with shape (B, L, 4).

        Returns:
            A DependencyMap instance with computed dependency map and reconstruction.
        """
        _check_sequence(sequence)

        # Upcast for good precision
        logits = logits.astype(np.float64)

        # Destructure into reference output, masked output and mutated output
        sequence_length = logits.shape[1]
        if logits.ndim != 3 or logits.shape[-1] != 4 or logits.shape[0] != 1 + 4 * sequence_length:
            raise ValueError(
                "Logits must have shape B x L x 4 where L is the sequence length and "
                "B = 1 + L + 3 * L (reference, masked, mutated)."
            )
        reference = logits[0]
        masked = logits[1 : 1 + sequence_length]
        mutated = logits[1 + sequence_length :].reshape(sequence_length, 3, sequence_length, 4)

        # Compute dependency map (log odds ratio)
        reference_log_odds = _logits_to_log_odds(reference)
        mutated_log_odds = _logits_to_log_odds(mutated)  # L x 3 x L x 4
        interaction_scores = mutated_log_odds - reference_log_odds[None, None, :, :]
        dependency_map = np.max(np.abs(interaction_scores), axis=(1, 3))  # L x L

        # Reconstruction (at masked position)
        reconstruction = softmax(
            masked[np.arange(sequence_length), np.arange(sequence_length)], axis=-1
        )

        return cls(sequence, dependency_map, reconstruction)

    @classmethod
    def compute_batched(
        cls: type[Self],
        sequence: str,
        tokenize_func: TokenizeFunc[TokenizedType],
        forward_func: ForwardFunc[TokenizedType],
        batch_size: int = 64,
        enable_progress_bar: bool = True,
        subset: tuple[int, int] | None = None,
    ) -> Self:
        """
        Compute dependency map and reconstruction in batches.

        This is the recommended way to use this library.
        Please refer to `examples/specieslm.py` for an example on how to use this function.

        Args:
            sequence: The input DNA sequence.
            tokenize_func: Function to tokenize the sequence.
            forward_func: Function to compute logits from tokenized batch.
            batch_size: Batch size for forward computation.
            enable_progress_bar: Whether to show a progress bar.
            subset: Optional tuple specifying the start and end indices for a subsequence.

        Returns:
            A DependencyMap instance with computed dependency map and reconstruction.
        """
        dataset = DependencyMap.make_dataset(sequence, tokenize_func, subset=subset)
        logits = []
        iterator = range(0, len(dataset), batch_size)
        if enable_progress_bar:
            iterator = tqdm(iterator)
        for batch_start in iterator:
            batch_end = min(len(dataset), batch_start + batch_size)
            batch_logits = forward_func(dataset[batch_start:batch_end])
            if subset is not None:
                batch_logits = batch_logits[:, subset[0] : subset[1], :]
            logits.append(batch_logits)
        logits = np.concatenate(logits, axis=0)
        if subset is not None:
            sequence = sequence[subset[0] : subset[1]]
        return cls.from_logits(sequence, logits)

    @classmethod
    def compute_batched_str(
        cls: type[Self],
        sequence: str,
        forward_func: ForwardFunc[str],
        batch_size: int = 64,
        enable_progress_bar: bool = True,
    ) -> Self:
        """
        Compute dependency map and reconstruction in batches using string tokenization.

        Args:
            sequence: The input DNA sequence.
            forward_func: Function to compute logits from string batch.
            batch_size: Batch size for forward computation.
            enable_progress_bar: Whether to show a progress bar.

        Returns:
            A DependencyMap instance with computed dependency map and reconstruction.
        """
        return cls.compute_batched(
            sequence,
            functools.partial(_str_tokenize_func, mask_char="X"),
            forward_func,
            batch_size,
            enable_progress_bar,
        )

    def plot(
        self,
        fig: go.Figure | None = None,
        row: int | None = None,
        col: int | None = None,
        xaxis_name: str | None = None,
        yaxis_name: str | None = None,
        axis_offset: int | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zero_diagonal: bool = True,
    ) -> go.Figure:
        """
        Plot the dependency map with sequence logos as a heatmap.

        This creates a plot with the dependency map as a heatmap and sequence logos on the axes.
        The logos show the sequence and the reconstruction by the DNALM.

        Args:
            fig: The figure to add the plot to. If None, a new figure is created.
            row: The row to add the plot to in the figure.
            col: The column to add the plot to in the figure.
            xaxis_name: The name of the x-axis.
            yaxis_name: The name of the y-axis.
            xaxis_offset: If provided, both axis will be offset by this amount.
            zmin: The minimum value for the heatmap color scale.
            zmax: The maximum value for the heatmap color scale.
            zero_diagonal: Whether to remove self-dependency in the heatmap.

        Note:
            If an existing figure is provided, one must specify the row and column to plot in.
            Additionally, one must specify the names of the x- and y-axes. For row=1, col=1, this
            would be "x" and "y", respectively. Otherwise, this will be "xN" and "yN", where N is
            the index of the axis.

        Returns:
            The figure with the plot.
        """
        if zero_diagonal:
            # Remove self-dependency
            dependency_map = np.copy(self.dependency_map)
            np.fill_diagonal(dependency_map, 0)
        else:
            dependency_map = self.dependency_map

        # Create sequence logos
        sequence_logo = SequenceLogo.from_sequence(self.sequence)
        reconstrution_logo = SequenceLogo.from_reconstruction(self.reconstruction)

        # Create the figure
        if fig is None:
            fig = make_subplots(rows=1, cols=1)
            row = 1
            col = 1
            xaxis_name = "x"
            yaxis_name = "y"
        elif row is None or col is None or xaxis_name is None or yaxis_name is None:
            raise ValueError(
                "If `fig` is provided, `row`, `col`, and `xaxis_name` must be provided as well."
            )

        # Dependency map
        if axis_offset is None:
            axis_offset = 0
        x = np.arange(axis_offset, axis_offset + dependency_map.shape[1])
        y = np.arange(axis_offset, axis_offset + dependency_map.shape[0])
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=dependency_map,
                zmin=zmin,
                zmax=zmax,
                colorscale=matplotlib_scale_as_plotly("coolwarm"),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(row=row, col=col, visible=False, scaleanchor="y", constrain="domain")
        fig.update_yaxes(
            row=row,
            col=col,
            visible=False,
            scaleanchor="x",
            constrain="domain",
            autorange="reversed",
        )

        # Sequence logos
        fig.add_layout_image(
            source=sequence_logo.to_svg(data_url=True),
            xref=xaxis_name,
            yref=f"{yaxis_name} domain",
            x=-0.5,
            y=1.0,
            sizex=dependency_map.shape[1],
            sizey=0.05,
            xanchor="left",
            yanchor="bottom",
            sizing="stretch",
        )
        fig.add_layout_image(
            source=sequence_logo.to_svg(data_url=True, orientation="west"),
            xref=f"{xaxis_name} domain",
            yref=yaxis_name,
            x=0.0,
            y=dependency_map.shape[0] - 0.5,
            sizex=0.05,
            sizey=dependency_map.shape[0],
            xanchor="right",
            yanchor="bottom",
            sizing="stretch",
        )
        fig.add_layout_image(
            source=reconstrution_logo.to_svg(data_url=True, orientation="south"),
            xref=xaxis_name,
            yref=f"{yaxis_name} domain",
            x=-0.5,
            y=0.0,
            sizex=dependency_map.shape[1],
            sizey=0.1,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
        )
        fig.add_layout_image(
            source=reconstrution_logo.to_svg(data_url=True, orientation="east"),
            xref=f"{xaxis_name} domain",
            yref=yaxis_name,
            x=1.0,
            y=-0.5,
            sizex=0.1,
            sizey=dependency_map.shape[0],
            xanchor="left",
            yanchor="top",
            sizing="stretch",
        )
        return fig


def _str_tokenize_func(sequence: str, mask: int | None, mask_char: str) -> str:
    """
    Tokenize a sequence by replacing a character at the mask position with mask_char.

    Args:
        sequence: The input DNA sequence.
        mask: The index to mask, or None for no masking.
        mask_char: The character to use for masking.

    Returns:
        The tokenized sequence as a string.
    """
    if mask is not None:
        return sequence[:mask] + mask_char + sequence[mask + 1 :]
    else:
        return sequence


def _check_sequence(sequence: str) -> None:
    """
    Check that the sequence contains only valid DNA bases (A, C, G, T).

    Args:
        sequence: The input DNA sequence.

    Raises:
        ValueError: If the sequence contains invalid characters.
    """
    if not all(nt in "ACGT" for nt in sequence):
        raise ValueError("Sequence must only contain A, C, G, and T.")


def _logits_to_log_odds(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to log odds ratios for each base.

    Args:
        logits: Model output logits.

    Returns:
        Log odds ratios as a numpy array.
    """
    log_prob = log_softmax(logits, axis=-1)  # log(p)
    log_prob_inverse = np.log1p(-np.exp(log_prob))  # log(1 - exp(log(p)))
    return log_prob - log_prob_inverse  # log(p / (1 - p))
