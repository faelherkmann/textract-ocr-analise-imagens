"""Mede quanto o pre-processamento recupera num documento degradado.

Hipotese: parte da falha do OCR no caso dificil vem da imagem, nao do
motor. Se for verdade, endireitar e normalizar contraste antes de enviar
deve melhorar o resultado sem trocar de ferramenta.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from baseline_tesseract import GABARITO, taxas_de_erro

RAIZ = Path(__file__).resolve().parent.parent
INPUTS = RAIZ / "inputs"
OUTPUTS = RAIZ / "outputs"


def remover_sombra(cinza: Image.Image, raio: int = 45) -> Image.Image:
    """Achata a iluminacao dividindo a imagem pelo fundo estimado.

    Um desfoque gaussiano de raio grande apaga o texto e preserva apenas
    a variacao lenta de luz, ou seja, a propria sombra. Dividir a imagem
    original por esse fundo cancela o gradiente e devolve um papel de
    brilho uniforme.

    Isso precisa vir ANTES da estimativa de inclinacao: com a sombra
    presente, o gradiente de luz domina a projecao e a estimativa de
    angulo vai para o extremo da busca em vez do valor correto.
    """
    fundo = cinza.filter(ImageFilter.GaussianBlur(raio))
    frente = np.asarray(cinza, dtype=np.float32)
    atras = np.asarray(fundo, dtype=np.float32) + 1e-6
    achatada = np.clip(frente / atras * 235, 0, 255).astype(np.uint8)
    return Image.fromarray(achatada)


def estimar_inclinacao(cinza: Image.Image, limite: float = 12.0) -> float:
    """Estima o angulo de inclinacao por projecao de perfil.

    Para cada angulo candidato, gira a mascara de tinta e soma os pixels
    escuros de cada linha. Com o texto na horizontal, a tinta se concentra
    em poucas faixas e a variancia da projecao chega ao maximo. O angulo
    que maximiza essa variancia e a correcao a aplicar.
    """
    arr = np.asarray(ImageOps.autocontrast(cinza, cutoff=1), dtype=np.uint8)

    # Mascara binaria de tinta. Trabalhar so com texto, e nao com niveis
    # de cinza, impede que o fundo residual influencie a projecao.
    tinta = ((arr < 160) * 255).astype(np.uint8)
    mascara = Image.fromarray(tinta)

    # Reduz para acelerar. A estimativa de angulo nao precisa de resolucao.
    escala = 900 / max(mascara.size)
    if escala < 1:
        novo = (int(mascara.width * escala), int(mascara.height * escala))
        mascara = mascara.resize(novo, Image.BILINEAR)

    melhor_angulo, melhor_score = 0.0, -1.0
    for angulo in np.arange(-limite, limite + 0.25, 0.25):
        girada = mascara.rotate(angulo, resample=Image.BILINEAR, fillcolor=0)
        projecao = np.asarray(girada, dtype=np.float32).sum(axis=1)
        score = float(np.var(projecao))
        if score > melhor_score:
            melhor_angulo, melhor_score = float(angulo), score

    return melhor_angulo


def preprocessar(origem: Path, destino: Path) -> float:
    """Achata a luz, endireita e amplia. Devolve o angulo aplicado."""
    cinza = Image.open(origem).convert("L")

    cinza = remover_sombra(cinza)
    angulo = estimar_inclinacao(cinza)
    cinza = cinza.rotate(angulo, resample=Image.BICUBIC, fillcolor=255)

    cinza = ImageOps.autocontrast(cinza, cutoff=1)

    # Amplia 2x. O Tesseract trabalha melhor perto de 300 dpi, e o
    # LANCZOS preserva melhor a borda das letras que o bilinear.
    # Nao binarizo: o Tesseract aplica o proprio Otsu internamente, e um
    # limiar global meu so destruiria os tracos mais finos.
    cinza = cinza.resize((cinza.width * 2, cinza.height * 2), Image.LANCZOS)
    cinza.save(destino)
    return angulo


def ocr(caminho: Path) -> str:
    return subprocess.run(
        ["tesseract", str(caminho), "stdout", "-l", "por"],
        capture_output=True, text=True, check=True,
    ).stdout


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    alvo = "03_caso_dificil.jpg"
    origem = INPUTS / alvo
    tratada = OUTPUTS / "03_caso_dificil_tratada.png"

    antes = taxas_de_erro(GABARITO[alvo], ocr(origem))
    angulo = preprocessar(origem, tratada)
    depois = taxas_de_erro(GABARITO[alvo], ocr(tratada))

    print(f"Angulo de correcao estimado: {angulo:+.1f} graus\n")
    print(f"{'metrica':<22}{'antes':>10}{'depois':>10}{'delta':>10}")
    for chave in ("cer", "wer", "palavras_lidas"):
        a, d = antes[chave], depois[chave]
        print(f"{chave:<22}{a:>10}{d:>10}{d - a:>+10.2f}")

    print("\n--- texto apos pre-processamento ---")
    print(ocr(tratada))


if __name__ == "__main__":
    main()
