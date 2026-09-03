"""Processamento de PDF multipagina com a API assincrona.

PDF com mais de uma pagina so e aceito pelas operacoes assincronas, que
exigem o arquivo hospedado no S3 e devolvem um JobId.
"""

from __future__ import annotations

import sys
import time

import boto3

REGIAO = "us-east-1"
textract = boto3.client("textract", region_name=REGIAO)


def iniciar_job(bucket: str, chave: str) -> str:
    """Dispara a extracao e devolve o identificador do job."""
    resposta = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": chave}}
    )
    return resposta["JobId"]


def aguardar(job_id: str, timeout: int = 300) -> None:
    """Espera o job concluir usando backoff exponencial.

    Polling e adequado para um laboratorio. Em producao, o caminho
    correto e configurar NotificationChannel com SNS e reagir ao evento
    numa Lambda, em vez de manter um processo bloqueado esperando.
    """
    espera = 2.0
    limite = time.time() + timeout

    while time.time() < limite:
        status = textract.get_document_text_detection(JobId=job_id)
        estado = status["JobStatus"]

        if estado == "SUCCEEDED":
            return
        if estado == "FAILED":
            raise RuntimeError(status.get("StatusMessage", "Job falhou"))

        time.sleep(espera)
        # Teto de 30s evita que a espera cresca indefinidamente
        # em documentos longos.
        espera = min(espera * 2, 30.0)

    raise TimeoutError(f"Job {job_id} excedeu {timeout} segundos")


def coletar_linhas(job_id: str) -> list[str]:
    """Recolhe todas as paginas do resultado.

    ATENCAO: esta e a armadilha silenciosa da API assincrona. Ignorar o
    NextToken nao lanca excecao alguma. O resultado simplesmente volta
    truncado, e o bug passa despercebido em testes com arquivo pequeno.
    """
    linhas: list[str] = []
    token: str | None = None

    while True:
        parametros = {"JobId": job_id}
        if token:
            parametros["NextToken"] = token

        pagina = textract.get_document_text_detection(**parametros)
        linhas.extend(
            bloco["Text"]
            for bloco in pagina["Blocks"]
            if bloco["BlockType"] == "LINE"
        )

        token = pagina.get("NextToken")
        if not token:
            return linhas


def extrair_pdf(bucket: str, chave: str) -> list[str]:
    job_id = iniciar_job(bucket, chave)
    print(f"Job iniciado: {job_id}")
    aguardar(job_id)
    return coletar_linhas(job_id)


def main() -> None:
    if len(sys.argv) < 3:
        print("uso: python async_pdf.py <bucket> <chave.pdf>")
        raise SystemExit(1)

    linhas = extrair_pdf(sys.argv[1], sys.argv[2])
    print(f"{len(linhas)} linhas extraidas\n")
    for linha in linhas:
        print(linha)


if __name__ == "__main__":
    main()
