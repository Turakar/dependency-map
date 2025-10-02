import numpy as np
import torch
from transformers import AutoTokenizer, BertForMaskedLM

from dependency_map import DependencyMap, DependencyMapOptions


def main() -> None:
    model_name = "johahi/specieslm-fungi-upstream-k1"
    proxy_species = "kazachstania_africana_cbs_2517_gca_000304475"
    reference_sequence = (
        "ATGTTAATTCTTTGAAATGAATACCACCTAATAAAACTATACATTTTCAAAATAACAAATTGTTATGAAACTAATAGCTAATTATCTTCTAGTGACAATAACCACTTTA"
        "CTAGAAATTATTAAAAAAATACCGCGTTGAAAGACGATATAAAGTAGGAGATTAATAAATTTATATTCATATTTTCTTCAATCTAATGAAATTGAAGCGCAAGGATTGA"
        "TTATGTGATAGGATGCGTGAGTAGTAATGCATGAAAAAGGAGGAAGACGTGATTATAATATATGATGTAAAATTTTGATTCCATTTTGCGGATTCCTGTATCCTCGAGG"
        "AAAGACCTCTGGCATATTATATAGGGATATTATTCCTTTACAAAAAATGGAATGAAAGAATCAAAACAAAATTGTCATGTTTGCACAATTGCGTATACCATTGCATATA"
        "ATTATGTCGAAAATCATCAATATGTATGGGAATGATCATTCCATTTTTTTCACACTGGCATAATAATTTGAATATAATAATAATGGATGATGGGCATTATTGGGAAATA"
        "AAACTATGAAATTTGCTTTTAGTACGTGTATCGATAAATATCTCTAATTTTCTCAAGATCTGTTCCTCGTGGCCCAATGGTCACGGCGTCTGGCTACGAACCAGAAGAT"
        "TCCAGGTTCAAGTCCTGGCGGGGAAGATTTTTTTTAACATTGAATAGAAATTACAAAGGTCTTTTATGCTATTTTAAAGGGCTATAGGCGGCATAATATTTGCTTTTAA"
        "CCTTATTGTCATGACATTCTAATCACAACACTTAGCCCTTCTTTTGTATTTTTCCAAACTTTTTCAAGTCTTAAAATTTTCCCAATCGAAAAAAATGGAAGATCCGAGA"
        "AGGTTTTATTGCCCTACGCTTAATCCCAAATTTTGCCACCATATAAAATGAGTACGAGCGATATAATCGGACAACTGAATAGAAGCTTCTGACCAAGTGATATCTTATT"
        "AATACAAATCTACTGTACGATG"
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = BertForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_token = tokenizer.special_tokens_map["mask_token"]
    sequence_tokens = torch.tensor([tokenizer.get_vocab()[c] for c in "ACGT"])

    def tokenize_func(sequence: str, mask: int | None) -> str:
        tokens = list(sequence)
        if mask is not None:
            tokens[mask] = mask_token
        return proxy_species + " " + " ".join(tokens)

    def forward_func(batch: list[str]) -> np.ndarray:
        with torch.inference_mode():
            x = tokenizer(batch, return_tensors="pt")
            logits = model(x["input_ids"].to(device)).logits
            return logits[:, 2:-1, sequence_tokens].cpu().numpy()

    dependency_map = DependencyMap.compute_batched(
        reference_sequence,
        tokenize_func,
        forward_func,
        options=DependencyMapOptions(subset=(850, 950)),
    )
    fig = dependency_map.plot()
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=30))
    fig.write_image("examples/dependency_map.svg")


if __name__ == "__main__":
    main()
