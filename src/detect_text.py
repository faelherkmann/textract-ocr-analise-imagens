"""OCR simples com DetectDocumentText.

Use quando o objetivo e apenas o texto corrido. E a operacao mais
barata do Textract e nao cobra por FeatureType adicional.
"""

from __future__ import annotations

import sys

import boto3

from utils import (
    carregar_resposta,
    desenhar_caixas,
    resumo_de_confianca,
    salvar_resposta,
)

REGIAO = "us-east-1"

# Cliente criado no nivel do modulo. O boto3 reaproveita a conexao
# entre chamadas, entao instanciar uma vez evita overhead de handshake.
textract = boto3.client("textract", region_name=REGIAO)


def extrair_texto(caminho_imagem: str, usar_cache: bool = True) -> dict:
    """Extrai linhas de texto de uma imagem local via API sincrona.

    Args:
        caminho_imagem: caminho para JPEG ou PNG. A API sincrona aceita
            os bytes direto no corpo da requisicao, sem precisar do S3.
        usar_cache: se True, reaproveita uma resposta ja salva em disco
            em vez de gerar uma nova chamada cobrada.
    """
    cache = f"outputs/{_nome_base(caminho_imagem)}_detect.json"

    resposta = carregar_resposta(cache) if usar_cache else None
    if resposta is None:
        with open(caminho_imagem, "rb") as arquivo:
            bytes_imagem = arquivo.read()

        # Chamada sincrona: responde em segundos, uma pagina por vez.
        # Documentos com varias paginas exigem a API assincrona.
        resposta = textract.detect_document_text(
            Document={"Bytes": bytes_imagem}
        )
        salvar_resposta(resposta, cache)

    return resposta


def linhas_de(resposta: dict) -> list[dict]:
    """Filtra os blocos LINE e normaliza os campos que interessam.

    O Textract devolve blocos PAGE, LINE e WORD. LINE ja agrupa as
    palavras na ordem de leitura, entao e o nivel mais util para texto
    corrido. WORD serve quando preciso da posicao individual.
    """
    return [
        {
            "texto": bloco["Text"],
            "confianca": round(bloco["Confidence"], 2),
            "caixa": bloco["Geometry"]["BoundingBox"],
        }
        for bloco in resposta["Blocks"]
        if bloco["BlockType"] == "LINE"
    ]


def _nome_base(caminho: str) -> str:
    return caminho.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def main() -> None:
    caminho = sys.argv[1] if len(sys.argv) > 1 else "inputs/01_lista_impressa.png"

    resposta = extrair_texto(caminho)
    linhas = linhas_de(resposta)

    for linha in linhas:
        marca = " " if linha["confianca"] >= 90 else "!"
        print(f'{marca} [{linha["confianca"]:>6.2f}%] {linha["texto"]}')

    print("\nResumo:", resumo_de_confianca(resposta["Blocks"]))

    saida = f"outputs/{_nome_base(caminho)}_anotada.png"
    desenhar_caixas(caminho, resposta["Blocks"], saida)
    print(f"Imagem anotada salva em {saida}")


if __name__ == "__main__":
    main()
