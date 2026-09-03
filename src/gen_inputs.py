"""Gera 3 imagens sinteticas de teste para o laboratorio do Amazon Textract."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = str(Path(__file__).resolve().parent.parent / "inputs")
os.makedirs(OUT, exist_ok=True)

GF = "/usr/share/fonts/truetype/google-fonts"
DJ = "/usr/share/fonts/truetype/dejavu"
LB = "/usr/share/fonts/truetype/liberation"

def font(path, size):
    return ImageFont.truetype(path, size)

REG = f"{GF}/Poppins-Regular.ttf"
BOLD = f"{GF}/Poppins-Bold.ttf"
MED = f"{GF}/Poppins-Medium.ttf"
ITAL = f"{GF}/Lora-Italic-Variable.ttf"
MONO = f"{DJ}/DejaVuSansMono.ttf"


# ---------------------------------------------------------------- imagem 1
def lista_impressa():
    W, H = 1240, 1754           # A4 a 150 dpi
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text((90, 90), "ESCOLA MUNICIPAL VALE VERDE", font=font(BOLD, 40), fill="black")
    d.text((90, 145), "Lista de Material Escolar 2026", font=font(MED, 30), fill=(60, 60, 60))
    d.text((90, 190), "Turma: 6o ano B      Turno: Manha", font=font(REG, 24), fill=(90, 90, 90))
    d.line((90, 240, W - 90, 240), fill=(180, 180, 180), width=3)

    itens = [
        ("04", "Caderno universitario 96 folhas"),
        ("02", "Bloco de papel milimetrado A4"),
        ("10", "Caneta esferografica azul"),
        ("05", "Caneta esferografica vermelha"),
        ("01", "Compasso escolar com grafite"),
        ("01", "Transferidor 180 graus"),
        ("02", "Lapis grafite HB numero 2"),
        ("01", "Borracha branca macia"),
        ("01", "Apontador com deposito"),
        ("01", "Tesoura sem ponta 13 cm"),
        ("02", "Cola bastao 20 gramas"),
        ("01", "Estojo de canetas hidrograficas"),
        ("01", "Regua transparente 30 cm"),
        ("01", "Dicionario de lingua portuguesa"),
        ("01", "Calculadora simples de 8 digitos"),
        ("03", "Pasta plastica com elastico"),
    ]

    y = 290
    for qtd, desc in itens:
        d.text((100, y), qtd, font=font(MED, 26), fill=(20, 20, 20))
        d.text((175, y), "x", font=font(REG, 26), fill=(140, 140, 140))
        d.text((215, y), desc, font=font(REG, 26), fill=(20, 20, 20))
        y += 52

    y += 40
    d.line((90, y, W - 90, y), fill=(180, 180, 180), width=2)
    y += 30
    d.text((90, y), "OBSERVACOES", font=font(BOLD, 26), fill="black")
    y += 45
    for linha in [
        "Todo o material deve ser identificado com o nome do aluno.",
        "A entrega ocorre na primeira semana letiva, na secretaria.",
        "Material de uso coletivo sera reposto no segundo semestre.",
    ]:
        d.text((90, y), linha, font=font(REG, 23), fill=(50, 50, 50))
        y += 40

    d.text((90, H - 90), "Documento sintetico gerado para teste de OCR",
           font=font(REG, 18), fill=(160, 160, 160))

    img.save(f"{OUT}/01_lista_impressa.png", dpi=(150, 150))
    print("ok 01_lista_impressa.png")


# ---------------------------------------------------------------- imagem 2
def formulario_tabela():
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # faixa de topo deixando explicito que o documento e sintetico
    d.rectangle((0, 0, W, 70), fill=(240, 200, 60))
    d.text((90, 22), "DOCUMENTO SINTETICO  .  AMOSTRA PARA TESTE DE OCR",
           font=font(BOLD, 24), fill=(60, 45, 0))

    d.text((90, 120), "PAPELARIA FICTICIA DEMONSTRACAO LTDA", font=font(BOLD, 34), fill="black")
    d.text((90, 168), "Pedido de Compra   No 2026-0042", font=font(MED, 27), fill=(70, 70, 70))
    d.line((90, 215, W - 90, 215), fill=(120, 120, 120), width=3)

    # pares chave e valor, alvo do FeatureType FORMS
    campos = [
        ("Razao social:", "Papelaria Ficticia Demonstracao LTDA"),
        ("Documento:", "00.000.000/0001-00"),
        ("Data de emissao:", "12/02/2026"),
        ("Vencimento:", "14/03/2026"),
        ("Condicao de pagamento:", "30 dias"),
        ("Responsavel:", "Setor de Compras"),
        ("Centro de custo:", "ADM-014"),
    ]
    y = 260
    for chave, valor in campos:
        d.text((100, y), chave, font=font(MED, 25), fill=(40, 40, 40))
        d.text((470, y), valor, font=font(REG, 25), fill=(20, 20, 20))
        y += 48

    # checkbox, vira SELECTION_ELEMENT na resposta do Textract
    y += 20
    d.text((100, y), "Entrega urgente:", font=font(MED, 25), fill=(40, 40, 40))
    d.rectangle((470, y + 4, 498, y + 32), outline=(40, 40, 40), width=3)
    d.line((476, y + 18, 484, y + 27), fill=(20, 20, 20), width=4)
    d.line((484, y + 27, 493, y + 10), fill=(20, 20, 20), width=4)
    d.text((520, y), "Sim", font=font(REG, 25), fill=(20, 20, 20))
    d.rectangle((620, y + 4, 648, y + 32), outline=(40, 40, 40), width=3)
    d.text((670, y), "Nao", font=font(REG, 25), fill=(20, 20, 20))

    # tabela, alvo do FeatureType TABLES
    y += 90
    d.text((90, y), "ITENS DO PEDIDO", font=font(BOLD, 27), fill="black")
    y += 55

    cols = [90, 620, 760, 950, 1150]
    header = ["Descricao", "Qtd", "Valor unit.", "Subtotal"]
    linhas = [
        ["Caderno universitario 96 fls", "40", "12,90", "516,00"],
        ["Caneta esferografica azul", "120", "2,50", "300,00"],
        ["Cola bastao 20 g", "60", "4,75", "285,00"],
        ["Tesoura sem ponta 13 cm", "35", "9,20", "322,00"],
        ["Regua acrilica 30 cm", "50", "3,40", "170,00"],
        ["Pasta com elastico", "80", "5,60", "448,00"],
    ]

    alt = 52
    d.rectangle((cols[0], y, cols[-1], y + alt), fill=(235, 235, 235))
    for i, texto in enumerate(header):
        d.text((cols[i] + 14, y + 12), texto, font=font(BOLD, 23), fill=(20, 20, 20))
    y += alt

    for linha in linhas:
        for i, texto in enumerate(linha):
            d.text((cols[i] + 14, y + 12), texto, font=font(REG, 23), fill=(20, 20, 20))
        y += alt

    topo = y - alt * (len(linhas) + 1)
    for x in cols:
        d.line((x, topo, x, y), fill=(120, 120, 120), width=2)
    for k in range(len(linhas) + 2):
        yy = topo + k * alt
        d.line((cols[0], yy, cols[-1], yy), fill=(120, 120, 120), width=2)

    y += 45
    d.text((760, y), "TOTAL GERAL", font=font(BOLD, 25), fill=(20, 20, 20))
    d.text((1000, y), "2.041,00", font=font(BOLD, 25), fill=(20, 20, 20))

    y += 90
    d.text((90, y), "Valores e empresa ficticios, criados apenas para exercitar",
           font=font(REG, 20), fill=(150, 150, 150))
    d.text((90, y + 30), "a extracao de campos e tabelas pelo Amazon Textract.",
           font=font(REG, 20), fill=(150, 150, 150))

    img.save(f"{OUT}/02_formulario_tabela.png", dpi=(150, 150))
    print("ok 02_formulario_tabela.png")


# ---------------------------------------------------------------- imagem 3
def caso_dificil():
    """Mesma classe de conteudo, degradada de proposito.

    Aplica rotacao, gradiente de sombra, desfoque, ruido e baixo contraste,
    que sao exatamente as condicoes de uma foto ruim tirada por celular.
    """
    W, H = 1240, 1600
    img = Image.new("RGB", (W, H), (252, 250, 244))
    d = ImageDraw.Draw(img)

    d.text((110, 120), "Anotacoes da reuniao", font=font(ITAL, 52), fill=(35, 35, 60))
    d.text((110, 200), "Almoxarifado  .  03/09/2026", font=font(ITAL, 32), fill=(70, 70, 90))

    notas = [
        "Confirmar entrega dos cadernos ate sexta.",
        "Fornecedor pediu reajuste de 8 por cento.",
        "Verificar saldo em estoque: 42 unidades.",
        "Cotacao alternativa com a Distribuidora Sul.",
        "Prazo de pagamento negociado: 30/60 dias.",
        "Pendente: assinatura do responsavel.",
        "Reuniao de acompanhamento em 17/09.",
    ]
    y = 300
    for nota in notas:
        d.text((130, y), "-", font=font(ITAL, 36), fill=(45, 45, 70))
        d.text((165, y), nota, font=font(ITAL, 36), fill=(45, 45, 70))
        y += 78

    d.text((130, y + 40), "Total estimado: R$ 2.041,00", font=font(ITAL, 40), fill=(120, 30, 30))

    # sombra diagonal, simula luz lateral
    sombra = Image.new("L", (W, H), 0)
    grad = np.tile(np.linspace(0, 150, W, dtype=np.uint8), (H, 1))
    diag = np.linspace(0, 60, H, dtype=np.uint8)[:, None]
    sombra = Image.fromarray(np.clip(grad + diag, 0, 255).astype(np.uint8))
    escuro = Image.new("RGB", (W, H), (25, 25, 35))
    img = Image.composite(escuro, img, sombra.point(lambda v: int(v * 0.55)))

    # rotacao e desfoque
    img = img.rotate(-6.5, expand=True, fillcolor=(210, 208, 202), resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(1.4))

    # ruido de sensor
    arr = np.array(img).astype(np.int16)
    arr += np.random.normal(0, 13, arr.shape).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # reducao de contraste, como foto subexposta
    arr = np.array(img).astype(np.float32)
    arr = arr * 0.78 + 34
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    img.save(f"{OUT}/03_caso_dificil.jpg", quality=88, optimize=True)
    print("ok 03_caso_dificil.jpg")


if __name__ == "__main__":
    np.random.seed(7)
    lista_impressa()
    formulario_tabela()
    caso_dificil()
