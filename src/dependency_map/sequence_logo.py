from typing import Self, Type

import drawsvg as draw
import numpy as np
from data_url import construct_data_url

_LETTER_A = '<path d="M 0.85423196,1.0000045 C 0.80929976,0.89726149 0.76436786,0.79451881 0.71943563,0.69177576 c -0.14785801,0 -0.29571565,0 -0.44357363,0 -0.0444096,0.10274305 -0.0888192,0.20548573 -0.13322884,0.30822874 -0.04754438,0 -0.09508881,0 -0.14263319,0 C 0.14576809,0.66667123 0.2915361,0.33333788 0.43730411,4.5440109e-6 c 0.0423198,0 0.0846396,0 0.12695909,0 C 0.70950884,0.33333788 0.8547544,0.66667123 0.99999997,1.0000045 c -0.0485893,0 -0.0971786,0 -0.14576801,0 z M 0.55172429,0.27894454 C 0.5318326,0.22981141 0.51435604,0.1799552 0.49843135,0.12971168 0.46305113,0.26633634 0.39870041,0.39472489 0.34561394,0.52626507 c -0.007576,0.0179792 -0.0151533,0.0359546 -0.0227297,0.0539346 0.11807733,0 0.23615462,0 0.35423195,0 -0.0417971,-0.10041825 -0.0835946,-0.2008369 -0.1253919,-0.30125513 z" style="fill:#00b200;stroke-width:0.11089" aria-label="A" />'
_LETTER_C = '<path d="m 0.63333337,0.10762961 q -0.21296279,0 -0.3351852,0.1049048 -0.12222222,0.10490483 -0.12222222,0.28746644 0,0.1811992 0.11296298,0.28746648 0.1148148,0.10490476 0.34259262,0.10490476 0.0870372,0 0.16481463,-0.0109002 0.0777778,-0.0109002 0.15185199,-0.0272467 V 0.96049253 Q 0.87407397,0.98092847 0.79444459,0.99046423 0.71666679,1 0.60740747,1 0.40555574,1 0.27037046,0.93869179 0.13518521,0.8773857 0.06666668,0.76430649 0,0.65122728 0,0.49863844 0,0.35149924 0.07222224,0.23978243 0.1462963,0.12670322 0.28888893,0.0640328 0.43148152,0 0.63518523,0 0.84444446,0 1,0.05722079 L 0.93333329,0.16076318 Q 0.8722221,0.14032724 0.7962963,0.12397848 0.72222209,0.10762973 0.63333337,0.10762961 Z" style="fill:#0000ff;stroke-width:0.0315195" aria-label="C" />'
_LETTER_G = '<path d="M 0.58347387,0.47275241 H 1 v 0.4768396 Q 0.9021923,0.97547802 0.80269808,0.9877392 0.70320388,1 0.57672844,1 0.38954469,1 0.26138268,0.94005437 0.1332209,0.87874727 0.0657673,0.76703057 0,0.65395145 0,0.50000038 0,0.34741171 0.07419901,0.235695 0.14839797,0.1239783 0.28667792,0.06267033 0.42664419,0 0.62394604,0 q 0.1011806,0 0.1905566,0.0149866 0.0910624,0.01498622 0.1686342,0.04223433 L 0.92580144,0.16348804 Q 0.86172034,0.14032744 0.78077605,0.12397844 0.70151825,0.10762974 0.6155147,0.10762974 q -0.21585147,0 -0.33726815,0.1049047 -0.11973018,0.10490471 -0.11973018,0.28746615 0,0.11580408 0.0455312,0.20572215 0.0472176,0.0885559 0.14671166,0.13896467 0.0994941,0.0490462 0.2613828,0.0490462 0.0792578,0 0.13490721,-0.006811 0.0556494,-0.006811 0.1011806,-0.0163503 v -0.288828 H 0.58347425 Z" style="fill:#ffa500;stroke-width:0.030078" aria-label="G" />'
_LETTER_T = '<path d="M 0.58504672,1 H 0.41682241 V 0.11064423 H 0 V 0 H 1 V 0.11064423 H 0.58504672 Z" style="fill:#ff0000;stroke-width:0.0321069" aria-label="T" />'

_ALPHABET_DNA = [_LETTER_A, _LETTER_C, _LETTER_G, _LETTER_T]


class SequenceLogo:
    """A builder class for sequence logos."""

    def __init__(self, scores: np.ndarray, highlight: np.ndarray | int | None = None):
        """Internal."""
        self.scores = scores
        if highlight is None:
            self.highlight = np.zeros(scores.shape[0], dtype=bool)
        elif isinstance(highlight, int):
            self.highlight = np.zeros(scores.shape[0], dtype=bool)
            self.highlight[highlight] = True
        else:
            self.highlight = highlight

    @classmethod
    def from_scores(
        cls: Type[Self],
        scores: np.ndarray,
        highlight: np.ndarray | int | None = None,
    ) -> Self:
        """Create a sequence logo from a matrix of scores.

        A score is proportional to the height of the corresponding letter in the sequence logo.

        Args:
            scores: The matrix of scores. The shape should be (nucleotides, 4).
                The columns should correspond to the nucleotides in the order A, C, G, T.

        Returns:
            A sequence logo based on the scores.
        """
        return cls(scores, highlight)

    @classmethod
    def from_sequence_scores(
        cls: Type[Self],
        sequence: str,
        scores: np.ndarray,
        highlight: np.ndarray | int | None = None,
    ) -> Self:
        """Create a sequence logo from a matrix of scores.

        A score is proportional to the height of the corresponding letter in the sequence logo.

        Args:
            scores: The matrix of scores. The shape should be (nucleotides,).

        Returns:
            A sequence logo based on the scores.
        """
        one_hot = np.array([[1 if nt == letter else 0 for nt in "ACGT"] for letter in sequence])
        scores_one_hot = one_hot * scores[:, None]
        return cls(scores_one_hot, highlight)

    @classmethod
    def from_reconstruction(
        cls: Type[Self],
        reconstruction: np.ndarray,
        highlight: np.ndarray | int | None = None,
    ) -> Self:
        """Create a sequence logo from a matrix of reconstructions.

        The information content of each nucleotide is computed as the entropy of the distribution of the nucleotides.
        For this, the background distribution is assumed to be uniform.

        Args:
            reconstruction: The matrix of reconstructions. The shape should be (nucleotides, 4).
                The columns should correspond to the nucleotides in the order A, C, G, T.

        Returns:
            A sequence logo based on the information content of the reconstructions.
        """
        information_content = -np.sum(reconstruction * (np.log2(reconstruction)), axis=1)
        background = np.array([0.25, 0.25, 0.25, 0.25])
        background_information_content = -np.sum(background * np.log2(background))
        scores = reconstruction * (background_information_content - information_content)[..., None]
        return cls(scores, highlight)

    @classmethod
    def from_sequence(
        cls: Type[Self], sequence: str, highlight: np.ndarray | int | None = None
    ) -> Self:
        """Create a sequence logo from a sequence.

        This sequence logo contains the letters of the sequence as a one-hot encoding, i.e.,
        there is only a single letter per position with height 1.

        Args:
            sequence: The sequence to convert to a sequence logo.

        Returns:
            A sequence logo based on the sequence.
        """
        scores = np.zeros((len(sequence), 4))
        for i, letter in enumerate(sequence):
            if letter == "A":
                scores[i, 0] = 1
            elif letter == "C":
                scores[i, 1] = 1
            elif letter == "G":
                scores[i, 2] = 1
            elif letter == "T":
                scores[i, 3] = 1
        return cls(scores, highlight)

    def draw(
        self,
        width_per_nucleotide: float | None = None,
        height: int | None = None,
        orientation: str | None = None,
    ) -> draw.Drawing:
        """Draw the sequence logo as a `drawsvg.Drawing`.

        Args:
            width_per_nucleotide: The width of each nucleotide in the sequence logo. Defaults to 10.
            height: The height of the sequence logo. Defaults to 50.
            orientation: The orientation of the sequence logo. One of "north", "west", "south", "east".
                The directions correspond to the viewing direction.
                Defaults to "north".

        Returns:
            The rendered sequence logo.
        """
        if width_per_nucleotide is None:
            width_per_nucleotide = 10.0
        if height is None:
            height = 50
        if orientation is None:
            orientation = "north"
        scores = self.scores
        total_width = width_per_nucleotide * scores.shape[0]
        if orientation == "north":
            d = draw.Drawing(total_width, height)
            g = draw.Group()
            d.append(g)
        elif orientation == "west":
            d = draw.Drawing(height, total_width)
            g = draw.Group(transform="rotate(-90) scale(-1, 1)")
            d.append(g)
        elif orientation == "south":
            d = draw.Drawing(total_width, height)
            g = draw.Group(transform=f"translate(0, {height}) scale(1, -1)")
            d.append(g)
        elif orientation == "east":
            d = draw.Drawing(height, total_width)
            g = draw.Group(transform=f"translate({height}, 0) rotate(90)")
            d.append(g)
        else:
            raise ValueError(f"Invalid orientation: {orientation}")

        low_end = min(0, scores.clip(max=0).sum(axis=-1).min())
        high_end = max(0, scores.clip(min=0).sum(axis=-1).max())
        zero_line = height - -low_end / (high_end - low_end) * height
        scale = height / (high_end - low_end)
        for i in range(scores.shape[0]):
            if self.highlight[i]:
                g.append(
                    draw.Rectangle(
                        i * width_per_nucleotide,
                        0,
                        width_per_nucleotide,
                        height,
                        fill="rgba(0, 0, 0, 0.1)",
                    )
                )
            abs_order = np.argsort(np.abs(scores[i]))
            positive_nt = [nt for nt in abs_order if scores[i, nt] > 0]
            positive_scores = [scores[i, nt] for nt in positive_nt]
            negative_nt = [nt for nt in abs_order if scores[i, nt] < 0]
            negative_scores = [-scores[i, nt] for nt in negative_nt]
            for j in range(len(positive_nt)):
                x = i * width_per_nucleotide
                w = width_per_nucleotide
                y = zero_line - sum(positive_scores[: j + 1]) * scale
                h = positive_scores[j] * scale
                g.append(
                    draw.Group(
                        [draw.Raw(_ALPHABET_DNA[positive_nt[j]])],
                        transform=f"translate({x}, {y}) scale({w}, {h})",
                    )
                )
            for j in range(len(negative_nt)):
                x = i * width_per_nucleotide
                w = width_per_nucleotide
                y = zero_line + sum(negative_scores[: j + 1]) * scale
                h = negative_scores[j] * scale
                g.append(
                    draw.Group(
                        [draw.Raw(_ALPHABET_DNA[negative_nt[j]])],
                        transform=f"translate({x}, {y}) scale({w}, -{h})",
                    )
                )
        return d

    def to_svg(
        self,
        width_per_nucleotide: float | None = None,
        height: int | None = None,
        orientation: str | None = None,
        data_url: bool = False,
    ) -> str:
        """Convert the sequence logo to an SVG string.

        Args:
            width_per_nucleotide: The width of each nucleotide in the sequence logo. Defaults to 10.
            height: The height of the sequence logo. Defaults to 50.
            orientation: The orientation of the sequence logo. One of "north", "west", "south", "east".
                The directions correspond to the viewing direction.
                Defaults to "north".
            data_url: Whether to return a data URL. Defaults to False. If True, return a URL of the form
                `data:image/svg+xml;base64,...`.

        Returns:
            The SVG string of the sequence logo or a data URL, respectively.
        """
        svg = self.draw(
            width_per_nucleotide=width_per_nucleotide, height=height, orientation=orientation
        ).as_svg()
        assert svg is not None
        if data_url:
            url = construct_data_url(
                mime_type="image/svg+xml",
                base64_encoded=True,
                data=svg.encode("utf-8"),
            )
            return url
        else:
            return svg
