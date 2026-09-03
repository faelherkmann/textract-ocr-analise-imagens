"""Baseline local com Tesseract, medido contra o texto verdadeiro.

Como os documentos de teste foram gerados por codigo, eu conheco
exatamente o texto correto de cada um. Isso permite medir o erro em vez
de estima-lo no olho, e serve de linha de base para comparar depois com
o Amazon Textract.

Metricas:
    CER  Character Error Rate, distancia de edicao normalizada por caractere
    WER  Word Error Rate, mesma ideia no nivel de palavra

Ambas sao "quanto menor, melhor". Zero significa transcricao perfeita.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
INPUTS = RAIZ / "inputs"
OUTPUTS = RAIZ / "outputs"


# ------------------------------------------------------- texto verdadeiro
GABARITO: dict[str, str] = {
    "01_lista_impressa.png": """
ESCOLA MUNICIPAL VALE VERDE
Lista de Material Escolar 2026
Turma: 6o ano B Turno: Manha
04 x Caderno universitario 96 folhas
02 x Bloco de papel milimetrado A4
10 x Caneta esferografica azul
05 x Caneta esferografica vermelha
01 x Compasso escolar com grafite
01 x Transferidor 180 graus
02 x Lapis grafite HB numero 2
01 x Borracha branca macia
01 x Apontador com deposito
01 x Tesoura sem ponta 13 cm
02 x Cola bastao 20 gramas
01 x Estojo de canetas hidrograficas
01 x Regua transparente 30 cm
01 x Dicionario de lingua portuguesa
01 x Calculadora simples de 8 digitos
03 x Pasta plastica com elastico
OBSERVACOES
Todo o material deve ser identificado com o nome do aluno.
A entrega ocorre na primeira semana letiva, na secretaria.
Material de uso coletivo sera reposto no segundo semestre.
Documento sintetico gerado para teste de OCR
""",
    "02_formulario_tabela.png": """
DOCUMENTO SINTETICO . AMOSTRA PARA TESTE DE OCR
PAPELARIA FICTICIA DEMONSTRACAO LTDA
Pedido de Compra No 2026-0042
Razao social: Papelaria Ficticia Demonstracao LTDA
Documento: 00.000.000/0001-00
Data de emissao: 12/02/2026
Vencimento: 14/03/2026
Condicao de pagamento: 30 dias
Responsavel: Setor de Compras
Centro de custo: ADM-014
Entrega urgente: Sim Nao
ITENS DO PEDIDO
Descricao Qtd Valor unit. Subtotal
Caderno universitario 96 fls 40 12,90 516,00
Caneta esferografica azul 120 2,50 300,00
Cola bastao 20 g 60 4,75 285,00
Tesoura sem ponta 13 cm 35 9,20 322,00
Regua acrilica 30 cm 50 3,40 170,00
Pasta com elastico 80 5,60 448,00
TOTAL GERAL 2.041,00
Valores e empresa ficticios, criados apenas para exercitar
a extracao de campos e tabelas pelo Amazon Textract.
""",
    "03_caso_dificil.jpg": """
Anotacoes da reuniao
Almoxarifado . 03/09/2026
- Confirmar entrega dos cadernos ate sexta.
- Fornecedor pediu reajuste de 8 por cento.
- Verificar saldo em estoque: 42 unidades.
- Cotacao alternativa com a Distribuidora Sul.
- Prazo de pagamento negociado: 30/60 dias.
- Pendente: assinatura do responsavel.
- Reuniao de acompanhamento em 17/09.
Total estimado: R$ 2.041,00
""",
}


# ------------------------------------------------------------ normalizacao
def normalizar(texto: str) -> str:
    """Remove acentos, pontuacao solta e espacos redundantes.

    Sem isso, a metrica pune diferencas irrelevantes de renderizacao em
    vez de medir erro real de reconhecimento.
    """
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^\w\s,./:%-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


# --------------------------------------------------------------- distancia
def levenshtein(a: list, b: list) -> int:
    """Distancia de edicao com uma unica linha de memoria.

    Guardar so a linha anterior baixa o espaco de O(n*m) para O(m), o que
    importa quando a comparacao e feita caractere a caractere numa pagina
    inteira.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        atual = [i]
        for j, cb in enumerate(b, start=1):
            insercao = atual[j - 1] + 1
            remocao = anterior[j] + 1
            substituicao = anterior[j - 1] + (ca != cb)
            atual.append(min(insercao, remocao, substituicao))
        anterior = atual

    return anterior[-1]


def taxas_de_erro(verdadeiro: str, lido: str) -> dict:
    v_norm, l_norm = normalizar(verdadeiro), normalizar(lido)
    v_pal, l_pal = v_norm.split(), l_norm.split()

    cer = levenshtein(list(v_norm), list(l_norm)) / max(len(v_norm), 1)
    wer = levenshtein(v_pal, l_pal) / max(len(v_pal), 1)

    return {
        "cer": round(cer * 100, 2),
        "wer": round(wer * 100, 2),
        "palavras_esperadas": len(v_pal),
        "palavras_lidas": len(l_pal),
    }


# ------------------------------------------------------------------ ocr
def rodar_tesseract(caminho: Path) -> tuple[str, list[dict]]:
    """Executa o Tesseract e devolve o texto e a confianca por palavra."""
    texto = subprocess.run(
        ["tesseract", str(caminho), "stdout", "-l", "por"],
        capture_output=True, text=True, check=True,
    ).stdout

    tsv = subprocess.run(
        ["tesseract", str(caminho), "stdout", "-l", "por", "tsv"],
        capture_output=True, text=True, check=True,
    ).stdout

    palavras = []
    leitor = csv.DictReader(tsv.splitlines(), delimiter="\t")
    for linha in leitor:
        conteudo = (linha.get("text") or "").strip()
        confianca = float(linha.get("conf") or -1)
        # conf == -1 marca blocos estruturais, nao palavras reconhecidas
        if conteudo and confianca >= 0:
            palavras.append({"texto": conteudo, "confianca": confianca})

    return texto, palavras


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    relatorio = {}

    for arquivo, verdadeiro in GABARITO.items():
        caminho = INPUTS / arquivo
        if not caminho.exists():
            print(f"ausente: {arquivo}")
            continue

        texto, palavras = rodar_tesseract(caminho)
        metricas = taxas_de_erro(verdadeiro, texto)

        confs = [p["confianca"] for p in palavras]
        metricas["confianca_media"] = round(sum(confs) / len(confs), 2) if confs else 0.0
        metricas["palavras_abaixo_de_70"] = sum(1 for c in confs if c < 70)

        relatorio[arquivo] = metricas
        (OUTPUTS / f"{caminho.stem}_tesseract.txt").write_text(texto, encoding="utf-8")

        print(f"\n=== {arquivo} ===")
        for chave, valor in metricas.items():
            print(f"  {chave:<24} {valor}")

    (OUTPUTS / "baseline_tesseract.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRelatorio salvo em outputs/baseline_tesseract.json")


if __name__ == "__main__":
    main()
