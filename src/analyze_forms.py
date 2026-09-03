"""Extracao de pares chave e valor e de tabelas com AnalyzeDocument.

Cada FeatureType solicitado e cobrado separadamente, entao peca apenas
o que for realmente usado. FORMS custa consideravelmente mais que TABLES.
"""

from __future__ import annotations

import sys

import boto3

from utils import (
    carregar_resposta,
    mapa_de_blocos,
    salvar_resposta,
    texto_do_bloco,
)

REGIAO = "us-east-1"
textract = boto3.client("textract", region_name=REGIAO)


def analisar(
    caminho_imagem: str,
    recursos: list[str] | None = None,
    usar_cache: bool = True,
) -> dict:
    """Chama AnalyzeDocument com os FeatureTypes escolhidos."""
    recursos = recursos or ["FORMS", "TABLES"]
    sufixo = "_".join(r.lower() for r in sorted(recursos))
    cache = f"outputs/{_nome_base(caminho_imagem)}_{sufixo}.json"

    resposta = carregar_resposta(cache) if usar_cache else None
    if resposta is None:
        with open(caminho_imagem, "rb") as arquivo:
            resposta = textract.analyze_document(
                Document={"Bytes": arquivo.read()},
                FeatureTypes=recursos,
            )
        salvar_resposta(resposta, cache)

    return resposta


def extrair_formulario(resposta: dict) -> dict[str, str]:
    """Devolve um dicionario {rotulo: valor} extraido do documento.

    Blocos KEY e VALUE compartilham o mesmo BlockType KEY_VALUE_SET.
    O que os distingue e o campo EntityTypes. A ligacao entre eles vem
    pela relacao de tipo VALUE, que parte da chave.
    """
    blocos = resposta["Blocks"]
    mapa = mapa_de_blocos(blocos)

    chaves = [
        b for b in blocos
        if b["BlockType"] == "KEY_VALUE_SET"
        and "KEY" in b.get("EntityTypes", [])
    ]

    resultado: dict[str, str] = {}
    for bloco_chave in chaves:
        rotulo = texto_do_bloco(bloco_chave, mapa)
        if not rotulo:
            continue

        valor = ""
        for relacao in bloco_chave.get("Relationships", []):
            if relacao["Type"] == "VALUE":
                for id_valor in relacao["Ids"]:
                    valor = texto_do_bloco(mapa[id_valor], mapa)

        # Remove o dois pontos final, comum em rotulos de formulario,
        # para que a chave fique utilizavel como nome de campo.
        resultado[rotulo.rstrip(":").strip()] = valor.strip()

    return resultado


def extrair_tabelas(resposta: dict) -> list[list[list[str]]]:
    """Converte blocos TABLE em matrizes de strings.

    Limitacao conhecida: celulas mescladas trazem RowSpan ou ColumnSpan
    maior que 1. Este codigo preenche apenas a celula ancora e deixa as
    demais vazias. Para documentos com merge, o valor precisa ser
    replicado ao longo do span.
    """
    blocos = resposta["Blocks"]
    mapa = mapa_de_blocos(blocos)
    tabelas: list[list[list[str]]] = []

    for bloco in blocos:
        if bloco["BlockType"] != "TABLE":
            continue

        celulas = []
        for relacao in bloco.get("Relationships", []):
            if relacao["Type"] == "CHILD":
                celulas.extend(mapa[i] for i in relacao["Ids"])

        celulas = [c for c in celulas if c["BlockType"] == "CELL"]
        if not celulas:
            continue

        # RowIndex e ColumnIndex comecam em 1, nao em 0.
        total_linhas = max(c["RowIndex"] for c in celulas)
        total_colunas = max(c["ColumnIndex"] for c in celulas)

        matriz = [
            ["" for _ in range(total_colunas)]
            for _ in range(total_linhas)
        ]
        for celula in celulas:
            linha = celula["RowIndex"] - 1
            coluna = celula["ColumnIndex"] - 1
            matriz[linha][coluna] = texto_do_bloco(celula, mapa)

        tabelas.append(matriz)

    return tabelas


def _nome_base(caminho: str) -> str:
    return caminho.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def main() -> None:
    caminho = sys.argv[1] if len(sys.argv) > 1 else "inputs/02_formulario_tabela.png"
    resposta = analisar(caminho)

    print("=== CAMPOS ===")
    for chave, valor in extrair_formulario(resposta).items():
        print(f"{chave:<28} => {valor}")

    print("\n=== TABELAS ===")
    for indice, tabela in enumerate(extrair_tabelas(resposta), start=1):
        print(f"\nTabela {indice}:")
        for linha in tabela:
            print(" | ".join(celula.ljust(28) for celula in linha))


if __name__ == "__main__":
    main()
