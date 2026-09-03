# Extração de Texto em Imagens: OCR local medido, e o que o Amazon Textract resolve

Desafio prático do Bootcamp **Nexa: Análise Avançada de Imagens e Texto com IA na AWS**, da [Digital Innovation One](https://www.dio.me).

![Tesseract](https://img.shields.io/badge/OCR-Tesseract%205.3-4B8BBE)
![AWS](https://img.shields.io/badge/AWS-Textract-FF9900?logo=amazonaws&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-green)

> ### Escopo: o que foi executado e o que não foi
>
> **Não tenho conta AWS**, então a parte do Textract deste repositório é **implementação de referência não executada**: o código está escrito e comentado, mas eu não rodei contra o serviço e portanto não reporto nenhum número dele.
>
> Em vez de parar aí, montei o experimento que **conseguia** executar: um baseline de OCR local com Tesseract, medido contra o texto verdadeiro dos documentos, para entender empiricamente onde o OCR quebra e o que exatamente um serviço gerenciado precisa resolver.
>
> Todos os números da seção 5 são medições reais, reproduzíveis com `python src/baseline_tesseract.py`. Nenhum resultado do Textract é reportado, porque nenhum foi obtido.

---

## Sumário

1. [A pergunta do experimento](#1-a-pergunta-do-experimento)
2. [Documentos de teste e por que são sintéticos](#2-documentos-de-teste-e-por-que-são-sintéticos)
3. [Metodologia](#3-metodologia)
4. [Como reproduzir](#4-como-reproduzir)
5. [Resultados medidos](#5-resultados-medidos)
6. [O que isso mostra sobre o Textract](#6-o-que-isso-mostra-sobre-o-textract)
7. [Implementação de referência do Textract](#7-implementação-de-referência-do-textract)
8. [Insights](#8-insights)
9. [Possibilidades de evolução](#9-possibilidades-de-evolução)
10. [Custos e limites do Textract](#10-custos-e-limites-do-textract)
11. [Estrutura do repositório](#11-estrutura-do-repositório)
12. [Referências](#12-referências)

---

## 1. A pergunta do experimento

O desafio pedia extrair texto de imagens com reconhecimento avançado. Sem acesso ao serviço pago, reformulei a pergunta para algo que eu pudesse responder com evidência:

> **Onde exatamente um OCR falha, e qual parte dessa falha é do motor de reconhecimento e qual é da imagem de entrada?**

Isso importa porque é a pergunta que decide a arquitetura. Se a falha vem da imagem, trocar de OCR não resolve e o dinheiro certo vai para pré-processamento. Se a falha vem do motor, aí sim vale pagar por um serviço melhor.

Testei três hipóteses:

| # | Hipótese | Como testei |
|---|---|---|
| H1 | OCR local resolve documento impresso limpo | Documento sintético de alta qualidade |
| H2 | Degradação de imagem derruba o OCR mais que a complexidade do texto | Mesmo tipo de conteúdo, com sombra, rotação, desfoque e ruído |
| H3 | Pré-processamento recupera o que a degradação tirou | Correção automática antes do OCR, medindo antes e depois |

---

## 2. Documentos de teste e por que são sintéticos

Os três documentos em `inputs/` foram **gerados por código** (`src/gen_inputs.py`), com empresa, valores e pessoas fictícios.

Foi escolha deliberada, por dois motivos:

**Privacidade.** Nota fiscal, boleto e documento de identidade são dados pessoais. Versionar isso em repositório público seria erro sério, e ainda por cima de LGPD.

**Controle experimental.** Como eu gerei os documentos, conheço o texto exato de cada um. Isso permite **medir** o erro em vez de estimar no olho. Sem gabarito, "funcionou bem" é opinião; com gabarito, vira número.

| Arquivo | O que exercita |
|---|---|
| `01_lista_impressa.png` | Linha de base. Texto impresso, alto contraste, alinhado. |
| `02_formulario_tabela.png` | Campos chave e valor separados por espaço horizontal, tabela de 4 colunas, checkbox. |
| `03_caso_dificil.jpg` | Rotação de 6,5°, gradiente de sombra, desfoque gaussiano, ruído de sensor e contraste reduzido. |

---

## 3. Metodologia

**Motor:** Tesseract 5.3.4 com pacote de idioma português.

**Métricas:** CER (Character Error Rate) e WER (Word Error Rate), ambas por distância de Levenshtein normalizada, com texto normalizado antes da comparação (sem acentos, minúsculas, pontuação solta removida, espaços colapsados). Normalizar evita punir diferenças de renderização em vez de erro real de reconhecimento.

Ambas são "quanto menor, melhor". Zero é transcrição perfeita.

```python
def levenshtein(a: list, b: list) -> int:
    """Distância de edição com uma única linha de memória.

    Guardar só a linha anterior baixa o espaço de O(n*m) para O(m), o que
    importa quando a comparação é caractere a caractere numa página inteira.
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
```

**Pipeline de pré-processamento testado em H3:**

1. Achatamento de iluminação por divisão pelo fundo estimado
2. Estimativa de inclinação por projeção de perfil
3. Rotação corretiva
4. Normalização de contraste
5. Ampliação 2x com LANCZOS

O passo 1 é o que destrava os demais:

```python
def remover_sombra(cinza: Image.Image, raio: int = 45) -> Image.Image:
    """Achata a iluminação dividindo a imagem pelo fundo estimado.

    Um desfoque gaussiano de raio grande apaga o texto e preserva apenas a
    variação lenta de luz, ou seja, a própria sombra. Dividir a imagem
    original por esse fundo cancela o gradiente.

    Isso precisa vir ANTES da estimativa de inclinação: com a sombra
    presente, o gradiente de luz domina a projeção e a estimativa de ângulo
    vai para o extremo da busca em vez do valor correto.
    """
    fundo = cinza.filter(ImageFilter.GaussianBlur(raio))
    frente = np.asarray(cinza, dtype=np.float32)
    atras = np.asarray(fundo, dtype=np.float32) + 1e-6
    achatada = np.clip(frente / atras * 235, 0, 255).astype(np.uint8)
    return Image.fromarray(achatada)
```

Descobri a ordem correta errando: na primeira versão eu estimava a inclinação antes de tratar a sombra, e o estimador retornava consistentemente **-12°**, exatamente o limite da busca, em vez dos +6,5° reais. O gradiente de luz produzia mais variância na projeção do que as próprias linhas de texto. Deixei registrado porque o sintoma (estimador colado no limite) é um bom indicador de que a métrica está medindo a coisa errada.

---

## 4. Como reproduzir

Não precisa de conta AWS nem cartão. Só Python e Tesseract.

```bash
# Ubuntu ou WSL
sudo apt install tesseract-ocr tesseract-ocr-por

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract tesseract-lang

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/gen_inputs.py            # regenera os documentos de teste
python src/baseline_tesseract.py    # mede CER e WER nos três
python src/preprocess_test.py       # mede o ganho do pré-processamento
```

---

## 5. Resultados medidos

### 5.1 Baseline, sem pré-processamento

| Documento | CER | WER | Palavras lidas / esperadas | Confiança média |
|---|---|---|---|---|
| `01_lista_impressa.png` | 6,56% | 9,52% | 140 / 147 | 90,1 |
| `02_formulario_tabela.png` | 6,50% | 7,69% | 109 / 117 | 89,2 |
| `03_caso_dificil.jpg` | **88,17%** | **92,98%** | **6 / 57** | 78,6 |

**H1 confirmada, com ressalva.** Documento impresso limpo sai com CER de 6,5%, mas os erros não são aleatórios. São confusões clássicas de glifo:

| Texto correto | Tesseract leu | Onde |
|---|---|---|
| `01` | `Ol` e `O1` | 5 ocorrências na lista |
| `6o ano` | `60 ano` | cabeçalho da turma |
| `50` | `bo` | coluna Qtd da tabela |

Repare no último: **um valor numérico numa coluna financeira lido como texto, sem erro nenhum sendo levantado.** O arquivo de saída não avisa. É a falha mais perigosa das três, porque passa silenciosa.

**H2 confirmada, e de forma mais brutal do que eu esperava.** No caso degradado o Tesseract não errou palavras, ele **desistiu**. Leu 6 de 57 palavras. A saída inteira foi:

```
Anotacoes da reuniao.
Almoxarifado : 03/09/2026
```

As sete linhas de conteúdo simplesmente não existiram na saída. E a confiança média reportada foi **78,6**, um número que passa longe de comunicar "eu perdi 89% do documento". Confiança média sobre o que foi lido não diz nada sobre o que não foi.

### 5.2 Com pré-processamento

| Documento | Ângulo detectado | CER antes | CER depois | WER antes | WER depois |
|---|---|---|---|---|---|
| `01_lista_impressa.png` | 0,00° | 6,56% | **2,39%** | 9,52% | **7,48%** |
| `02_formulario_tabela.png` | 0,00° | 6,50% | **42,65%** | 7,69% | **46,15%** |
| `03_caso_dificil.jpg` | +6,50° | 88,17% | **0,27%** | 92,98% | **1,75%** |

**H3 confirmada no caso degradado, e refutada nos limpos.**

No documento difícil o ganho foi quase total: CER de 88,17% para **0,27%**, e as 57 palavras recuperadas. O estimador acertou o ângulo em **+6,50°**, contra os -6,5° que eu apliquei ao gerar a imagem. Transcrição praticamente perfeita:

```
Anotacoes da reuniao
Almoxarifado - 03/09/2026
- Confirmar entrega dos cadernos ate sexta.
- Fornecedor pediu reajuste de 8 por cento.
- Verificar saldo em estoque: 42 unidades.
- Cotacao alternativa com a Distribuidora Sul.
- Prazo de pagamento negociado: 30/60 dias.
- Pendente: assinatura do responsavel.
- Reuniao de acompanhamento em 17/09.
Total estimado: R$ 2.041,00
```

**O motor nunca foi o gargalo. A imagem era.** O mesmo Tesseract que tinha lido 10% do documento leu 100% dele depois de endireitar e achatar a luz.

Mas o formulário limpo **piorou seis vezes**, de 6,50% para 42,65%. Fui olhar o que aconteceu: o texto continuou sendo reconhecido, mas a **ordem de leitura quebrou**. Rótulos e valores, que antes saíam na mesma linha, se separaram em dois blocos:

```
Razao social:              →   Papelaria Ficticia Demonstracao LTDA
Documento:                     00.000.000/0001-00
Data de emissao:               12/02/2026
```

virou

```
Razao social:
Documento:
Data de emissao:
...
Papelaria Ficticia Demonstracao LTDA
00.000.000/0001-00
12/02/2026
```

E a tabela numérica virou `era [out` e `o am uso`.

A conclusão prática: **pré-processamento não é bom por si só, é bom condicionalmente.** Aplicar às cegas degrada documento que já estava bom. Um pipeline sério precisa estimar a qualidade da entrada primeiro e só então decidir se trata.

---

## 6. O que isso mostra sobre o Textract

Aqui está o valor real do experimento: ele produziu evidência para uma comparação que, sem medir, seria só uma tabela de alegações.

| Problema medido | Como o Tesseract se comportou | O que o Textract oferece |
|---|---|---|
| Ordem de leitura em layout de duas colunas | Separou rótulos dos valores após o pré-processamento | `AnalyzeDocument` com `FORMS` devolve o par ligado por `Relationships`, sem depender de proximidade visual |
| Valor numérico lido errado sem aviso (`50` → `bo`) | Nada sinalizou o erro | `Confidence` por bloco permite limiar e fila de revisão humana |
| Estrutura de tabela | Vira texto separado por espaço; reconstruir exige heurística | `TABLES` devolve `RowIndex` e `ColumnIndex` por célula |
| Checkbox | Lido como `[]`, sem semântica | `SELECTION_ELEMENT` com `SelectionStatus` explícito |
| Documento degradado | Perdeu 89% do conteúdo | Não medido. Sem conta AWS, não afirmo nada. |

E a contrapartida, igualmente medida: **para o documento impresso limpo, o Tesseract já entrega 93,5% de acerto de caractere de graça e offline.** Pagar US$ 50 por mil páginas em `AnalyzeDocument FORMS` só se justifica quando a estrutura importa. Para OCR simples de alto volume, o serviço gerenciado é dinheiro jogado fora.

A linha divisória que o experimento desenhou: **o Textract não é um OCR melhor, é um extrator de estrutura.** Comprar pelo motivo errado é caro.

---

## 7. Implementação de referência do Textract

> ⚠️ Código escrito e comentado, **não executado**. Sem conta AWS, não posso validar contra o serviço.

O conceito central, e o que mais me custou entender lendo a documentação: a resposta do Textract **não é uma lista plana de textos**. É um grafo. Cada bloco tem um `Id` e uma lista `Relationships` apontando para outros blocos. `KEY_VALUE_SET` e `CELL` não guardam texto próprio, só referências. Quem lê a resposta linearmente conclui que a API não funcionou.

```python
def texto_do_bloco(bloco: dict, mapa: dict[str, dict]) -> str:
    """Reconstrói o texto de um bloco seguindo seus filhos CHILD."""
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
                # Checkbox. O Textract informa apenas o estado, então
                # represento marcado e desmarcado de forma explícita.
                marcado = filho.get("SelectionStatus") == "SELECTED"
                partes.append("[X]" if marcado else "[ ]")

    return " ".join(partes)
```

```python
def extrair_formulario(resposta: dict) -> dict[str, str]:
    """Devolve um dicionário {rótulo: valor} extraído do documento."""
    blocos = resposta["Blocks"]
    mapa = mapa_de_blocos(blocos)

    # Blocos KEY e VALUE compartilham o mesmo BlockType KEY_VALUE_SET.
    # O que os distingue é o campo EntityTypes.
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

        # A relação de tipo VALUE parte da chave e aponta para o valor.
        valor = ""
        for relacao in bloco_chave.get("Relationships", []):
            if relacao["Type"] == "VALUE":
                for id_valor in relacao["Ids"]:
                    valor = texto_do_bloco(mapa[id_valor], mapa)

        resultado[rotulo.rstrip(":").strip()] = valor.strip()

    return resultado
```

Duas armadilhas que documentei em `src/`:

**Células mescladas.** `RowSpan` e `ColumnSpan` maiores que 1 não são tratados em `analyze_forms.py`. O código preenche só a célula âncora. Deixei registrado em vez de esconder.

**Paginação silenciosa.** Na API assíncrona, ignorar `NextToken` não lança exceção. O resultado volta truncado. É o tipo de bug que passa em teste com arquivo pequeno e falha com o documento do cliente.

```python
def coletar_linhas(job_id: str) -> list[str]:
    linhas: list[str] = []
    token: str | None = None

    while True:
        parametros = {"JobId": job_id}
        if token:
            parametros["NextToken"] = token

        pagina = textract.get_document_text_detection(**parametros)
        linhas.extend(
            b["Text"] for b in pagina["Blocks"] if b["BlockType"] == "LINE"
        )

        token = pagina.get("NextToken")
        if not token:
            return linhas
```

O `.gitignore` bloqueia `.env`, e a política IAM sugerida em `docs/iam-policy.json` concede só as cinco ações do Textract mais leitura e escrita num bucket específico. Chave estática nunca entra em código; em produção o caminho é IAM Role.

---

## 8. Insights

**1. Medir mudou a conclusão.** Eu entrei achando que ia comparar qualidade de OCR. Saí com outra coisa: a diferença entre Tesseract e Textract não é acurácia de reconhecimento, é **estrutura**. No documento limpo o Tesseract lê quase tudo. O que ele não faz é dizer que `12/02/2026` é o valor de `Data de emissão`.

**2. O erro perigoso é o silencioso.** `50` lido como `bo` numa coluna de quantidade não gera exceção, não baixa a confiança média de forma visível, e contamina o dado a jusante. É por isso que `Confidence` por elemento é recurso de produto, não detalhe técnico: permite montar fluxo híbrido, aceitando automaticamente acima de um limiar e mandando o resto para revisão humana.

**3. Confiança média mente.** No caso degradado o Tesseract reportou 78,6 de confiança média tendo perdido 89% do documento. A média é calculada sobre o que ele leu, não sobre o que existia. Métrica de cobertura precisa vir separada de métrica de qualidade.

**4. A imagem pesa mais que o motor.** CER de 88,17% para 0,27% no mesmo Tesseract, só arrumando a entrada. Antes de trocar de ferramenta ou de pagar por serviço, vale medir quanto do problema está na captura.

**5. Pré-processamento cego faz mal.** O mesmo pipeline que salvou o documento difícil piorou o limpo em seis vezes, quebrando a ordem de leitura. Um sistema real precisa estimar a qualidade da entrada e decidir se trata, em vez de aplicar sempre.

**6. Estimador colado no limite da busca é sintoma, não ruído.** Quando a inclinação estimada deu exatamente -12° (o limite), o problema não era o intervalo, era a métrica medindo sombra em vez de texto. Valor de saída no extremo do domínio quase sempre significa que a função objetivo está errada.

**7. Custo por página muda a forma de desenvolver.** Como cada chamada ao Textract é cobrada, o `utils.py` tem cache em disco antes de ter qualquer outra coisa: salvar o JSON bruto e iterar o parser em cima do arquivo salvo, sem regerar chamada. Restrição de custo virou decisão de arquitetura.

---

## 9. Possibilidades de evolução

Do experimento:

- [ ] Estimador de qualidade de entrada, para aplicar pré-processamento condicionalmente
- [ ] Métrica de cobertura separada da de acurácia, já que confiança média esconde omissão
- [ ] Comparar com EasyOCR e PaddleOCR, que são locais e gratuitos, para isolar motor de imagem
- [ ] Binarização Sauvola por janela, em vez de limiar global
- [ ] Executar a implementação do Textract quando houver acesso, e preencher a coluna vazia da seção 6

Aplicações que a extração estruturada viabiliza:

- **Automação de contas a pagar.** Nota chega por e-mail, `AnalyzeExpense` extrai fornecedor, valor e vencimento, o sistema lança e sinaliza divergência.
- **Digitalização de arquivo físico.** Contratos e laudos escaneados viram base pesquisável.
- **Onboarding com validação de documento.** `AnalyzeID` extrai os campos e o Rekognition compara a foto com uma selfie.
- **Acessibilidade.** Material impresso convertido para leitores de tela.
- **Encadeamento com Comprehend ou Bedrock** para extrair entidades e responder perguntas sobre o documento já transcrito.

---

## 10. Custos e limites do Textract

Consultado em setembro de 2026, região Oregon. Valores mudam por região e ao longo do tempo; confira a [página oficial](https://aws.amazon.com/pt/textract/pricing/).

| Operação | Nível gratuito (3 meses) | Depois, por mil páginas |
|---|---|---|
| `DetectDocumentText` | 1.000 páginas/mês | US$ 1,50 |
| `AnalyzeDocument` Tables | 100 páginas/mês | US$ 15,00 |
| `AnalyzeDocument` Forms | 100 páginas/mês | US$ 50,00 |
| `AnalyzeExpense` | 100 páginas/mês | US$ 10,00 |
| `AnalyzeID` | 100 páginas/mês | US$ 25,00 |

Cada `FeatureType` é cobrado separadamente: pedir `FORMS` e `TABLES` juntos soma os dois preços. A diferença de mais de trinta vezes entre `DetectDocumentText` e `AnalyzeDocument FORMS` é a razão de a seção 6 existir: escolher a operação errada é caro, nos dois sentidos.

Outros limites: a API síncrona tem teto de tamanho no corpo da requisição (o `03_caso_dificil` precisou virar JPEG por esse motivo, em PNG dava 5,4 MB); PDF multipágina só é aceito nas operações assíncronas; o serviço não está em todas as regiões.

---

## 11. Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
├── src/
│   ├── gen_inputs.py           # gera os documentos sintéticos de teste
│   ├── baseline_tesseract.py   # OCR local + CER/WER contra o gabarito
│   ├── preprocess_test.py      # achatamento de luz, deskew e medição do ganho
│   ├── utils.py                # travessia do grafo de blocos do Textract
│   ├── detect_text.py          # [referência] OCR simples
│   ├── analyze_forms.py        # [referência] chave e valor + tabelas
│   └── async_pdf.py            # [referência] PDF multipágina via S3
├── inputs/                     # documentos sintéticos
├── outputs/                    # transcrições e métricas (ignorado no git)
└── docs/                       # política IAM e imagens do README
```

Arquivos marcados como `[referência]` implementam o Textract e não foram executados.

---

## 12. Referências

- [O que é o Amazon Textract](https://docs.aws.amazon.com/pt_br/textract/latest/dg/what-is.html)
- [Analisando texto do documento](https://docs.aws.amazon.com/pt_br/textract/latest/dg/analyzing-document-text.html)
- [Referência da API Textract no boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/textract.html)
- [Preços do Amazon Textract](https://aws.amazon.com/pt/textract/pricing/)
- [Documentação do Tesseract](https://tesseract-ocr.github.io/)
- [Repositório base do bootcamp](https://github.com/digitalinnovationone/nexa-analise-avancada-de-imagens-e-texto-com-ia-na-aws)
- [Bootcamp Nexa na DIO](https://www.dio.me/bootcamp/analise-avancada-imagens-texto-ia-aws)

---

## Licença

[MIT](LICENSE).

---

**Autor:** Raphael Herkmann
[GitHub](https://github.com/faelherkmann) · [LinkedIn](https://www.linkedin.com/in/raphael-herkmann/)

Desenvolvido durante o Bootcamp Nexa da [Digital Innovation One](https://www.dio.me).
