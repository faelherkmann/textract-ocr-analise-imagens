"""Funcoes auxiliares compartilhadas pelos scripts do laboratorio.

O ponto central deste modulo e a travessia do grafo de blocos que o
Amazon Textract devolve. A resposta nao e uma lista plana de textos:
cada bloco carrega um Id e uma lista Relationships apontando para
outros blocos. Sem percorrer esse grafo, formularios e tabelas parecem
vazios.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


# --------------------------------------------------------------- grafo
def mapa_de_blocos(blocos: list[dict]) -> dict[str, dict]:
    """Indexa os blocos por Id para resolver relacionamentos em O(1).

    Sem esse indice, cada busca por um Id filho seria uma varredura
    linear na lista inteira, o que vira O(n^2) em documentos densos.
    """
    return {bloco["Id"]: bloco for bloco in blocos}


def texto_do_bloco(bloco: dict, mapa: dict[str, dict]) -> str:
    """Reconstroi o texto de um bloco seguindo seus filhos CHILD.

    Blocos estruturais (KEY_VALUE_SET, CELL) nao guardam texto proprio.
    Eles referenciam blocos WORD e SELECTION_ELEMENT pela relacao CHILD.
    """
    partes: list[str] = []

    for relacao in bloco.get("Relationships", []):
        if relacao["Type"] != "CHILD":
            continue

        for id_filho in relacao["Ids"]:
            filho = mapa.get(id_filho)
            if filho is None:
                continue

            if filho["BlockType"] == "WORD":
                partes.append(filho["Text"])

            elif filho["BlockType"] == "SELECTION_ELEMENT":
                # Checkbox. O Textract informa apenas o estado,
                # entao represento marcado e desmarcado de forma explicita.
                marcado = filho.get("SelectionStatus") == "SELECTED"
                partes.append("[X]" if marcado else "[ ]")

    return " ".join(partes)


# --------------------------------------------------------------- cache
def salvar_resposta(resposta: dict, destino: str) -> None:
    """Persiste a resposta bruta da API em disco.

    Cada chamada ao Textract e cobrada por pagina. Salvar o JSON bruto
    permite desenvolver e depurar o parser quantas vezes for preciso
    sem gerar custo novo nem esperar a rede.
    """
    caminho = Path(destino)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(resposta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def carregar_resposta(origem: str) -> dict | None:
    """Le uma resposta salva, ou devolve None se ainda nao existir."""
    caminho = Path(origem)
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


# --------------------------------------------------------- visualizacao
def desenhar_caixas(
    caminho_imagem: str,
    blocos: list[dict],
    destino: str,
    tipo: str = "LINE",
    limiar: float = 90.0,
) -> None:
    """Desenha as bounding boxes sobre a imagem original.

    Verde marca blocos acima do limiar de confianca, vermelho marca os
    abaixo. E a forma mais rapida de auditar visualmente onde o modelo
    hesitou, em vez de ler numeros num JSON.
    """
    imagem = Image.open(caminho_imagem).convert("RGB")
    largura, altura = imagem.size
    desenho = ImageDraw.Draw(imagem)

    for bloco in blocos:
        if bloco.get("BlockType") != tipo:
            continue

        # BoundingBox vem normalizada entre 0 e 1, relativa a pagina.
        # Para desenhar em pixels, multiplico pelas dimensoes reais.
        caixa = bloco["Geometry"]["BoundingBox"]
        esquerda = caixa["Left"] * largura
        topo = caixa["Top"] * altura
        direita = esquerda + caixa["Width"] * largura
        base = topo + caixa["Height"] * altura

        confianca = bloco.get("Confidence", 100.0)
        cor = (0, 170, 60) if confianca >= limiar else (220, 40, 40)

        desenho.rectangle((esquerda, topo, direita, base), outline=cor, width=3)

    os.makedirs(Path(destino).parent, exist_ok=True)
    imagem.save(destino)


def resumo_de_confianca(blocos: list[dict], tipo: str = "LINE") -> dict:
    """Calcula estatisticas de confianca para relatar no README."""
    valores = [
        b["Confidence"] for b in blocos
        if b.get("BlockType") == tipo and "Confidence" in b
    ]
    if not valores:
        return {"blocos": 0}

    return {
        "blocos": len(valores),
        "media": round(sum(valores) / len(valores), 2),
        "minima": round(min(valores), 2),
        "maxima": round(max(valores), 2),
        "abaixo_de_90": sum(1 for v in valores if v < 90),
    }
